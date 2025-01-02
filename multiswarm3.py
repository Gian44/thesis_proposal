import itertools
import math
import random
from deap import base, creator, tools
from initialize_population2 import assign_courses
from functools import partial
import time

# Initialize globals for course and room data, which will be set in main()
courses = []
rooms = []
curricula = []
data = []
room_map = {}
reverse_room_map = {}

# Define the fitness and particle classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Particle", list, fitness=creator.FitnessMin, speed=list, best=None, bestfit=None, is_quantum=False)
creator.create("Swarm", list, best=None, bestfit=None)

# Generate a particle (schedule) using Graph-Based Heuristic
def generate(pclass, course_order):
    schedule = None
    
    # Keep trying to assign courses until a valid schedule is generated
    while not schedule:
        schedule = assign_courses()  # Generate the initial timetable
    
    particle = []

    for day, periods in schedule.items():
        for period, rooms in periods.items():
            for room, course_id in rooms.items():
                if course_id != -1:  # Skip empty slots
                    particle.append({
                        "day": day,
                        "period": period,
                        "room_id": room,
                        "course_id": course_id
                    })

    # Sort the particle list based on the order of courses in `course_order`
    particle.sort(key=lambda x: course_order.index(x["course_id"]))
    
    return pclass(particle)


# Evaluate a particle (schedule) for soft constraint penalties
def calculate_room_capacity_penalty(particle, rooms, courses):
    room_capacity_penalty = 0
    for entry in particle:
        room = entry["room_id"]
        course_id = entry["course_id"]

        # Skip empty slots
        if course_id == -1:
            continue

        room_details = next((r for r in rooms if r["id"] == room), None)
        if room_details and any(course["id"] == course_id for course in courses):
            course = next(course for course in courses if course["id"] == course_id)
            if course["num_students"] > room_details["capacity"]:
                room_capacity_penalty += (course["num_students"] - room_details["capacity"])
    return room_capacity_penalty

def calculate_min_days_violation_penalty(particle, courses, course_assignments):
    min_days_violation_penalty = 0

    # Calculate penalty for each course
    for course in courses:
        course_id = course["id"]
        assignments = course_assignments[course_id]
        days_used = {assignment["day"] for assignment in assignments}
        missing_days = max(0, course["min_days"] - len(days_used))
        min_days_violation_penalty += 5 * missing_days
    return min_days_violation_penalty

def calculate_room_stability_penalty(particle, courses, course_assignments):
    room_stability_penalty = 0

    # Calculate penalty for each course
    for course in courses:
        course_id = course["id"]
        assignments = course_assignments[course_id]
        rooms_used = {assignment["room_id"] for assignment in assignments}
        room_stability_penalty += max(0, len(rooms_used) - 1)  # Each extra room adds 1 penalty point
    return room_stability_penalty

def calculate_curriculum_compactness_penalty(particle, curricula, courses, course_assignments):
    curriculum_compactness_penalty = 0

    # Calculate curriculum compactness penalty
    for curriculum in curricula:
        curriculum_courses = curriculum["courses"]
        curriculum_assignments = [
            assignment
            for course_id in curriculum_courses
            for assignment in course_assignments[course_id]
        ]

        # Group assignments by day
        assignments_by_day = {}
        for assignment in curriculum_assignments:
            day = assignment["day"]
            if day not in assignments_by_day:
                assignments_by_day[day] = []
            assignments_by_day[day].append(assignment)

        # Check for non-adjacent lectures in the same day
        for day, day_assignments in assignments_by_day.items():
            day_assignments.sort(key=lambda x: x["period"])
            for i in range(len(day_assignments)):
                current = day_assignments[i]
                is_isolated = True
                if i > 0:  # Check previous
                    previous = day_assignments[i - 1]
                    if current["period"] == previous["period"] + 1:
                        is_isolated = False
                if i < len(day_assignments) - 1:  # Check next
                    next_lecture = day_assignments[i + 1]
                    if current["period"] + 1 == next_lecture["period"]:
                        is_isolated = False
                if is_isolated:
                    curriculum_compactness_penalty += 2  # Each isolated lecture adds 2 points
    return curriculum_compactness_penalty

def evaluate_schedule(particle, rooms, courses, curricula, constraints):
    """
    Evaluates the fitness of a particle by calculating the total penalty
    for soft constraints.
    """
    # Track course assignments
    course_assignments = {course["id"]: [] for course in courses}
    for entry in particle:
        if entry["course_id"] != -1:
            course_assignments[entry["course_id"]].append({"day": entry["day"], "period": entry["period"], "room_id": entry["room_id"]})

    room_capacity_penalty = calculate_room_capacity_penalty(particle, rooms, courses)
    min_days_violation_penalty = calculate_min_days_violation_penalty(particle, courses, course_assignments)
    room_stability_penalty = calculate_room_stability_penalty(particle, courses, course_assignments)
    curriculum_compactness_penalty = calculate_curriculum_compactness_penalty(particle, curricula, courses, course_assignments)

    # Calculate total penalty by summing all individual penalties
    total_penalty = (
        room_capacity_penalty +
        min_days_violation_penalty +
        room_stability_penalty +
        curriculum_compactness_penalty
    )

    # Return the total penalty as a tuple
    return (total_penalty,)

toolbox = base.Toolbox()

def updateParticle(data, particle, personal_best, global_best, chi, c1, c2, constraints):
    """
    Updates the particle using PSO velocity formula.
    Implements moves and swaps for Room, Day, Timeslot systematically.
    Each move selects a random course and ensures feasibility after each operation.
    """    

    days = data["num_days"]
    periods = data["periods_per_day"]
    rooms = data["rooms"]
    original_penalty = particle.fitness.values[0]     # Get original penalty

    # Function to select a random entry
    def select_random_entry():
        index = random.randint(0, len(particle) - 1)
        return index, particle[index]

    # Function to find corresponding personal and global best entries
    def get_best_entries(index):
        personal_best_entry = personal_best[index] if personal_best else None
        global_best_entry = global_best[index] if global_best else None
        return personal_best_entry, global_best_entry
    
    # Function to get random new values for day, period, and room
    def get_random_new_values():
        new_day = random.randint(0, days - 1)
        new_period = random.randint(0, periods - 1)
        new_room = random.choice(rooms)["id"]
        return new_day, new_period, new_room

    # Function to apply velocity-based update
    def calculate_new_values(entry, p_best_entry, g_best_entry):
        r1, r2 = random.random(), random.random()  # Random coefficients
        current_day = entry["day"]
        current_period = entry["period"]
        current_room_index = room_map[entry["room_id"]]

        # Personal and global best values for velocity calculation
        p_best_day = p_best_entry["day"] if p_best_entry else current_day
        g_best_day = g_best_entry["day"] if g_best_entry else current_day
        p_best_period = p_best_entry["period"] if p_best_entry else current_period
        g_best_period = g_best_entry["period"] if g_best_entry else current_period
        p_best_room_index = room_map[p_best_entry["room_id"]] if p_best_entry else current_room_index
        g_best_room_index = room_map[g_best_entry["room_id"]] if g_best_entry else current_room_index

        # Calculate velocity components
        velocity_day = chi * (c1 * r1 * (p_best_day - current_day) + c2 * r2 * (g_best_day - current_day))
        velocity_period = chi * (c1 * r1 * (p_best_period - current_period) + c2 * r2 * (g_best_period - current_period))
        velocity_room = chi * (c1 * r1 * (p_best_room_index - current_room_index) + c2 * r2 * (g_best_room_index - current_room_index))

        # Apply updates with modulo clamping
        new_day = (current_day + round(velocity_day)) % days
        new_period = (current_period + round(velocity_period)) % periods
        new_room_index = (current_room_index + round(velocity_room)) % len(rooms)
        new_room = reverse_room_map[new_room_index]
        
        return new_day, new_period, new_room

    # Moves and swaps
    moves = [
        "Room, Day, Timeslot",
        "Day, Timeslot",
        "Day",
        "Timeslot",
        "Room and Day",
        "Room and Timeslot",
        "Room",
    ]
    local_best = True if particle == global_best else False
    for move in moves:
        # MOVE PHASE
        i, entry = select_random_entry()  # Select a random assignment
        personal_best_entry, global_best_entry = get_best_entries(i)
        original_state = (entry["day"], entry["period"], entry["room_id"])

        if local_best:
            new_day, new_period, new_room = get_random_new_values()
        else:
            new_day, new_period, new_room = calculate_new_values(entry, personal_best_entry, global_best_entry)
        
        # Apply move
        if move == "Room, Day, Timeslot":
            entry["day"], entry["period"], entry["room_id"] = new_day, new_period, new_room

        elif move == "Day, Timeslot":
            entry["day"], entry["period"] = new_day, new_period

        elif move == "Day":
            entry["day"] = new_day

        elif move == "Timeslot":
            entry["period"] = new_period

        elif move == "Room and Day":
            entry["day"], entry["room_id"] = new_day, new_room

        elif move == "Room and Timeslot":
            entry["period"], entry["room_id"] = new_period, new_room

        elif move == "Room":
            entry["room_id"] = new_room

        # Check feasibility and penalty, revert if necessary
        new_penalty = evaluate_schedule(particle, rooms, courses, curricula, constraints)[0]
        if not is_feasible(particle, constraints, courses, curricula) or new_penalty >= original_penalty:
            entry["day"], entry["period"], entry["room_id"] = original_state
        else:
            print(f"{move} move successful. Penalty improved from {original_penalty} to {new_penalty}.")
            original_penalty = new_penalty

        # SWAP PHASE
        swap_index, entry = select_random_entry()  # Ensure a different course for swap
        swap_personal_best_entry, swap_global_best_entry = get_best_entries(swap_index)
        swap_original_state = (entry["day"], entry["period"], entry["room_id"])

        # Use calculate_new_values to determine the new day, period, and room

        if local_best:
            swap_new_day, swap_new_period, swap_new_room = get_random_new_values()
        else:
            swap_new_day, swap_new_period, swap_new_room = calculate_new_values(
                entry, swap_personal_best_entry, swap_global_best_entry
            )

        # Find the corresponding entry in the particle based on the new values
        swap_target_entry = next(
            (e for e in particle if e["day"] == swap_new_day and e["period"] == swap_new_period and e["room_id"] == swap_new_room),
            None
        )

        # Save the original state of the target entry (if exists)
        if swap_target_entry:
            swap_new_state = (swap_target_entry["day"], swap_target_entry["period"], swap_target_entry["room_id"])
        else:
            swap_new_state = None

        # Apply swap based on the move type
        if move == "Room, Day, Timeslot" and swap_target_entry:
            entry["day"], entry["period"], entry["room_id"], swap_target_entry["day"], swap_target_entry["period"], swap_target_entry["room_id"] = (
                swap_target_entry["day"], swap_target_entry["period"], swap_target_entry["room_id"],
                entry["day"], entry["period"], entry["room_id"]
            )
        elif move == "Day, Timeslot" and swap_target_entry:
            entry["day"], entry["period"], swap_target_entry["day"], swap_target_entry["period"] = (
                swap_target_entry["day"], swap_target_entry["period"],
                entry["day"], entry["period"]
            )
        elif move == "Day" and swap_target_entry:
            entry["day"], swap_target_entry["day"] = swap_target_entry["day"], entry["day"]
        elif move == "Timeslot" and swap_target_entry:
            entry["period"], swap_target_entry["period"] = swap_target_entry["period"], entry["period"]
        elif move == "Room and Day" and swap_target_entry:
            entry["day"], entry["room_id"], swap_target_entry["day"], swap_target_entry["room_id"] = (
                swap_target_entry["day"], swap_target_entry["room_id"],
                entry["day"], entry["room_id"]
            )
        elif move == "Room and Timeslot" and swap_target_entry:
            entry["period"], entry["room_id"], swap_target_entry["period"], swap_target_entry["room_id"] = (
                swap_target_entry["period"], swap_target_entry["room_id"],
                entry["period"], entry["room_id"]
            )
        elif move == "Room" and swap_target_entry:
            entry["room_id"], swap_target_entry["room_id"] = swap_target_entry["room_id"], entry["room_id"]

        # Check feasibility and penalty after swap
        swap_penalty = evaluate_schedule(particle, rooms, courses, curricula, constraints)[0]
        if not is_feasible(particle, constraints, courses, curricula) or swap_penalty >= original_penalty:
            # Revert swap if not feasible or penalty is worse
            entry["day"], entry["period"], entry["room_id"] = swap_original_state
            if swap_target_entry:
                swap_target_entry["day"], swap_target_entry["period"], swap_target_entry["room_id"] = swap_new_state
        else:
            print(f"{move} swap successful. Penalty improved from {original_penalty} to {swap_penalty}.")
            original_penalty = swap_penalty

def is_feasible(schedule, constraints, courses, curricula):
    """
    Check if the entire schedule adheres to all HARD constraints.
    
    Args:
        schedule (list): The schedule (particle) to check.
        constraints (list): Hard constraints.
        courses (list): Course details, including number of students and teachers.
        curricula (list): Curricula details, including associated courses.

    Returns:
        bool: True if the schedule satisfies all HARD constraints, False otherwise.
    """
    # Track room assignments by day, period, and room ID (hashmap)
    room_assignments = {}
    # Track course assignments by course ID (hashmap)
    course_assignments = {}
    # Track teacher conflicts (using sets to track course/day/period)
    teacher_conflicts = {}

    # Initialize dictionaries for room and course assignments
    for entry in schedule:
        day = entry["day"]
        period = entry["period"]
        room = entry["room_id"]
        course_id = entry["course_id"]

        # Initialize room assignments for the day and period
        if (day, period) not in room_assignments:
            room_assignments[(day, period)] = {}
        
        # Check if the room is already assigned
        if room in room_assignments[(day, period)]:
            return False  # H2 violation

        # Assign room to the current course at the given day and period
        room_assignments[(day, period)][room] = course_id

        # Track course assignments for the course ID
        if course_id not in course_assignments:
            course_assignments[course_id] = set()
        
        if (day, period) in course_assignments[course_id]:
            return False  # H1 violation
        course_assignments[course_id].add((day, period))

        # Check for teacher conflict (ensure one teacher isn't assigned to multiple courses at the same time)
        teacher = get_teacher(courses, course_id)
        if teacher:
            if (teacher, day, period) in teacher_conflicts:
                return False  # Teacher conflict
            teacher_conflicts[(teacher, day, period)] = course_id

    # Curriculum conflict: Check if courses from the same curriculum are scheduled at the same time
    for curriculum in curricula:
        curriculum_courses = curriculum["courses"]
        curriculum_assignments = [
            entry for entry in schedule if entry["course_id"] in curriculum_courses
        ]

        # Group assignments by day and period
        assignments_by_day_period = {}
        for entry in curriculum_assignments:
            day_period = (entry["day"], entry["period"])
            if day_period not in assignments_by_day_period:
                assignments_by_day_period[day_period] = []
            assignments_by_day_period[day_period].append(entry["course_id"])

        # Check if more than one course from the curriculum is scheduled in the same time slot
        for day_period, courses_in_slot in assignments_by_day_period.items():
            if len(courses_in_slot) > 1:  # More than one course in the same slot
                return False  # H3 Violation

    # Unavailability Constraints
    for constraint in constraints:
        for entry in schedule:
            if (
                constraint["course"] == entry["course_id"]
                and constraint["day"] == entry["day"]
                and constraint["period"] == entry["period"]
            ):
                return False  # H4 violation

    # If no conflicts were found, the schedule is feasible
    return True

def get_teacher(courses, course_id):
    """
    Retrieve the teacher for a given course ID.
    """
    for course in courses:
        if course["id"] == course_id:
            return course["teacher"]
    return None

def convertQuantum(swarm, rcloud, centre, constraints, courses, curricula, rooms, days, periods):
    for part in swarm:
        # Skip reinitializing the best particle
        if part == swarm.best:
            continue

        # Generate new positions around the best-found position (centre) for each course assignment
        for entry, best_entry in zip(part, centre):
            best_room_index = room_map[best_entry["room_id"]]

            # Apply Gaussian random movements
            new_day = round(best_entry["day"] + rcloud * random.gauss(0, 1))
            new_period = round(best_entry["period"] + rcloud * random.gauss(0, 1))
            new_room_index = round(best_room_index + rcloud * random.gauss(0, 1))

            # Apply modulo-based clamping for cyclic wrapping
            new_day %= days  # Replace with the actual number of days in the problem
            new_period %= periods    # Replace with the actual number of periods in the problem
            new_room_index %= len(rooms)

            # Convert index back to room ID
            new_room = reverse_room_map[new_room_index]
            
            # Temporarily assign new values to this particle's entry
            original_entry = (entry["day"], entry["period"], entry["room_id"])
            entry["day"], entry["period"], entry["room_id"] = new_day, new_period, new_room

            if is_feasible(part, constraints, courses, curricula):
                continue
            else:
                entry["day"], entry["period"], entry["room_id"] = original_entry


        # Update fitness after reinitialization
        part.fitness.values = toolbox.evaluate(part)
        part.best = None

# Main loop to simulate the Multi-Swarm Particle Swarm Optimization
def main(data, max_iterations=500, verbose=True):
    global courses, rooms, curricula, room_map, reverse_room_map
    courses = data["courses"]
    rooms = data["rooms"]
    curricula = data["curricula"]
    constraints = data["constraints"]
    days = data["num_days"]
    periods = data["periods_per_day"]
    room_map = {room['id']: i for i, room in enumerate(rooms)}
    reverse_room_map = {i: room['id'] for i, room in enumerate(rooms)}
    lectures = 0

    course_order = [course['id'] for course in courses]
    for course in courses:
        lectures += course["num_lectures"]

    toolbox.register("particle", partial(generate, course_order=course_order), creator.Particle)
    toolbox.register("swarm", tools.initRepeat, creator.Swarm, toolbox.particle)
    toolbox.register(
        "evaluate",
        partial(evaluate_schedule, rooms=rooms, courses=courses, curricula=curricula, constraints=constraints)
    )

    NSWARMS = 1
    NPARTICLES = 5
    NEXCESS = 3
    RCLOUD = 0.5
    NDIM = 3
    BOUNDS = len(rooms) * days * periods

    
    start_time = time.time()
    population = [toolbox.swarm(n=NPARTICLES) for _ in range(NSWARMS)]
    chi, c1, c2 = 0.729, 1, 1

    # Initialization
    best_global_fitness = float('inf')
    global_best_particle = None
    best_global_particle_idx = None
    initial_fitness_values = []
    last_global_best_update = -1
    init_flags = [False] * NSWARMS  # Init flags for randomization markers

    for swarm in population:
        swarm.best = None
        swarm.bestfit = creator.FitnessMin((float('inf'),))
        swarm.no_improvement_iters = 0

        for part in swarm:
            part.fitness.values = toolbox.evaluate(part)
            part.best = toolbox.clone(part)
            part.bestfit = creator.FitnessMin(part.fitness.values)
            if swarm.best is None or part.fitness.values < swarm.bestfit.values:
                swarm.best = toolbox.clone(part)
                swarm.bestfit.values = part.fitness.values

        if global_best_particle is None or swarm.bestfit.values[0] < best_global_fitness:
            best_global_fitness = swarm.bestfit.values[0]
            global_best_particle = toolbox.clone(swarm.best)

    for swarm in population:
        for part in swarm:
            fitness = part.fitness.values[0]
            initial_fitness_values.append(fitness)

    print("\nInitial Fitness Values Before Optimization:")
    for i, fitness in enumerate(initial_fitness_values):
        print(f"Particle {i + 1}: Fitness = {fitness:.2f}")

    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}/{max_iterations}")
        rexcl = (BOUNDS / len(population))** (1.0 / NDIM)
        print("Rexcl: ", rexcl) 
        print("Total Swarms: ", len(population))

        # Anti-Convergence
        print("Anti-Convergence check")
        all_converged = True
        worst_swarm_idx = None
        worst_swarm_fitness = float('-inf') # Initial fitness

        for i, swarm in enumerate(population):
            for p1, p2 in itertools.combinations(swarm, 2):
                distance = math.sqrt(
                    sum(
                        ((entry1["day"] - entry2["day"]) / days) ** 2 +
                        ((entry1["period"] - entry2["period"]) / periods) ** 2 +
                        ((room_map[entry1["room_id"]] - room_map[entry2["room_id"]]) / len(rooms)) ** 2
                        for entry1, entry2 in zip(p1, p2)
                    )
                )

                if distance > 2 * rexcl:
                    all_converged = False
                    print("Not all have converged yet")
                    break

            if all_converged and swarm.bestfit.values[0] > worst_swarm_fitness:
                worst_swarm_fitness = swarm.bestfit.values[0]
                worst_swarm_idx = i
                swarm.bestfit.values[0]
                print("Index: ", worst_swarm_idx)
                print("Fitness: ", worst_swarm_fitness)
        
        if all_converged and worst_swarm_idx is not None:
            print(f"Randomizing worst swarm: {worst_swarm_idx}")
            init_flags[worst_swarm_idx] = True

        # Adding swarms
        if all_converged and len(population) < NSWARMS + NEXCESS:
            new_swarm = toolbox.swarm(n=NPARTICLES)
            new_swarm.best = None
            new_swarm.bestfit = creator.FitnessMin((float('inf'),))
            new_swarm.no_improvement_iters = 0

            # Log the fitness of the new swarm
            print("\nNew Swarm Added:")
            for part in new_swarm:
                part.fitness.values = toolbox.evaluate(part)
                part.best = toolbox.clone(part)
                part.bestfit = creator.FitnessMin(part.fitness.values)
                if new_swarm.best is None or part.fitness < new_swarm.bestfit:
                    new_swarm.best = toolbox.clone(part)
                    new_swarm.bestfit.values = part.fitness.values
                # Log the fitness
                print(f"New Particle Fitness: {part.fitness.values[0]:.2f}")
                initial_fitness_values.append(part.fitness.values[0])

            population.append(new_swarm)
            init_flags.append(False) 

        # Exclusion
        print("Exclusion check")
        reinit_swarms = set()
        for s1, s2 in itertools.combinations(range(len(population)), 2):
            if population[s1].best and population[s2].best and not (s1 in reinit_swarms or s2 in reinit_swarms):
                distance = math.sqrt(
                    sum(
                        ((entry1["day"] - entry2["day"]) / days) ** 2 +
                        ((entry1["period"] - entry2["period"]) / periods) ** 2 +
                        ((room_map[entry1["room_id"]] - room_map[entry2["room_id"]]) / len(rooms)) ** 2
                        for entry1, entry2 in zip(population[s1].best, population[s2].best)
                    )
                )
                if distance < rexcl:
                    reinit_swarms.add(s1 if population[s1].bestfit <= population[s2].bestfit else s2)

        for s in reinit_swarms:
            print(f"Reinitializing swarm: {s}")
            init_flags[s] = True

        # Update and Randomize Particles
        for i, swarm in enumerate(population):
            if init_flags[i]:
                print(f"Swarm: {i+1}")
                convertQuantum(swarm, RCLOUD, swarm.best, constraints, courses, curricula, rooms, days, periods)
                init_flags[i] = False
                for j, part in enumerate(swarm):
                    print("Particle "+ str((5*i)+(j+1)) + " (Fitness: "+ str(part.fitness.values[0]) + ")")
                    if swarm.best is None or part.fitness.values < swarm.bestfit.values:
                        swarm.best = toolbox.clone(part)
                        swarm.bestfit.values = part.fitness.values
                print(f"Swarm has been reinitialized. Swarm bestfit is now {swarm.bestfit}.")
            else:
                for j, part in enumerate(swarm):
                    print("Particle "+ str((5*i)+(j+1)) + " (Fitness: "+ str(part.fitness.values[0]) + ")")
                    prev_pos = toolbox.clone(part)

                    if part == swarm.best:
                        print("This is the local best particle. Fitness: ", swarm.best.fitness)
                        if swarm.no_improvement_iters > 10:
                            print("Swarm needs improvement")
                            updateParticle(data, part, part.best, swarm.best, chi, c1, c2, constraints)
                        else:
                            continue  # Skip the rest of the loop for this particle
                    else:
                        updateParticle(data, part, part.best, swarm.best, chi, c1, c2, constraints)
                     
                     # Re-evaluate the fitness after updating the particle
                    if prev_pos != part:
                        part.fitness.values = toolbox.evaluate(part)
                    """ 
                    # Print current best in swarm and the current part fitness
                    print("Swarm bestfit before comparison:", swarm.bestfit.values)
                    print("Particle bestfit value: ", part.bestfit.values)
                    print("Current particle fitness:", part.fitness.values)
                    """
                    if part.bestfit is None or part.fitness.values[0] < part.bestfit.values[0]:
                        part.best = toolbox.clone(part)
                        part.bestfit.values = part.fitness.values
                        print("Updated part bestfit:", part.bestfit.values)

                    if part.fitness.values[0] < swarm.bestfit.values[0]:
                        swarm.best = toolbox.clone(part)
                        swarm.bestfit.values = part.fitness.values
                        swarm.no_improvement_iters = 0
                        print("****************UPDATED SWARM BESTFIT WITH NEW BEST PARTICLE****************")
                    else:
                        swarm.no_improvement_iters += 1
                        print("No improvement in swarm bestfit.")

        best_fitness_in_population = min(swarm.bestfit.values[0] for swarm in population if swarm.bestfit.values)
        print("Best fitness: ", best_fitness_in_population)
        if best_fitness_in_population < best_global_fitness:
            for swarm_idx, swarm in enumerate(population):
                for particle_idx, particle in enumerate(swarm):
                    if particle.fitness.values[0] < best_global_fitness:
                        best_global_fitness = particle.fitness.values[0]
                        global_best_particle = toolbox.clone(particle)
                        best_global_particle_idx = (swarm_idx, particle_idx)
                        last_global_best_update = iteration
                        print("##############GLOBAL BEST FITNESS UPDATED##############")
                        print(f"Global best updated at iteration {iteration + 1} by swarm {swarm_idx + 1}, particle {particle_idx + 1}")

            # Re-evaluate personal bests for particles
            for swarm in population:
                for particle in swarm:
                    # Check if current particle's fitness is better than its personal best
                    if particle.fitness < particle.bestfit:
                        particle.best = toolbox.clone(particle)
                        particle.bestfit = particle.fitness

        # Stop if the fitness meets the target of 0 or less
        if best_global_fitness <= 0:
            print(f"\nStopping early as target fitness of 0 was reached: {best_global_fitness}")
            break

    # End the timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    # Determine the best particle across all swarms based on bestfit values if available
    valid_swarms = [swarm for swarm in population if swarm.best is not None and swarm.bestfit is not None]

    print("\nOptimization Completed.")

    print("\nInitial Fitness Values Before Optimization:")
    for i, fitness in enumerate(initial_fitness_values):
        print(f"Particle {i + 1}: Fitness = {fitness:.2f}")

    particle_origin = (5*best_global_particle_idx[0]) + best_global_particle_idx[1] + 1
    if valid_swarms:
        # Find the swarm with the minimum bestfit value
        best_swarm = min(valid_swarms, key=lambda s: s.bestfit.values[0])
        final_best_schedule = [entry for entry in best_swarm.best]
        print("\nFinal Best Solution Found (Fitness):", best_swarm.bestfit.values[0])
        print(f"\nOptimization completed in {elapsed_time:.2f} seconds.")
    else:
        # Fallback if no swarm has a valid best (highly unlikely with this setup)
        final_best_schedule = None
        print("\nNo solution found.")

    print(f"The last global best was updated at iteration {last_global_best_update + 1}")
    if best_global_particle_idx:
        print(f"\nBest solution found by particle: ", particle_origin)
    else:
        print("\nNo valid best solution found.")

    return final_best_schedule