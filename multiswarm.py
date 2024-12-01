import itertools
import math
import random
import copy
from joblib import Parallel, delayed
from deap import base, creator, tools
from initialize_population2 import assign_courses
from functools import partial

# Initialize globals for course and room data, which will be set in main()
courses = []
rooms = []
curricula = []

# Define the fitness and particle classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Particle", list, fitness=creator.FitnessMin, speed=list, best=None, bestfit=None)
creator.create("Swarm", list, best=None, bestfit=None)

# Generate a particle (schedule) using Graph Based Heuristic
def generate(pclass):
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
    return pclass(particle)

# Evaluate a particle (schedule) for soft constraint penalties
def evaluate_schedule(particle, rooms, courses, curricula, constraints):
    """
    Evaluates the fitness of a particle by calculating the total penalty
    for soft constraints.

    Args:
        particle (list): The schedule (particle) to evaluate.
        rooms (list): List of rooms with their capacities.
        courses (list): List of courses with their details.
        curricula (list): List of curricula and their courses.
        constraints (list): Hard constraints.

    Returns:
        tuple: A single fitness value representing the total penalty for all soft constraints.
    """

    # Initialize penalties
    room_capacity_penalty = 0
    min_days_violation_penalty = 0
    curriculum_compactness_penalty = 0
    room_stability_penalty = 0

    # Convert particle into a timetable-like structure
    timetable = {}
    for entry in particle:
        day = entry["day"]
        period = entry["period"]
        room = entry["room_id"]
        course_id = entry["course_id"]

        if day not in timetable:
            timetable[day] = {}
        if period not in timetable[day]:
            timetable[day][period] = {}
        timetable[day][period][room] = course_id

    # Track course assignments and days used
    course_assignments = {course["id"]: [] for course in courses}
    for day, periods in timetable.items():
        for period, rooms_in_period in periods.items():
            for room, course_id in rooms_in_period.items():
                if course_id == -1:  # Skip empty slots
                    continue
                course_assignments[course_id].append({"day": day, "period": period, "room_id": room})

    # Evaluate penalties for each course
    for course in courses:
        course_id = course["id"]
        assignments = course_assignments[course_id]

        # Room Capacity Penalty
        for assignment in assignments:
            room = assignment["room_id"]
            room_details = next((r for r in rooms if r["id"] == room), None)
            if room_details and course["num_students"] > room_details["capacity"]:
                room_capacity_penalty += (course["num_students"] - room_details["capacity"])

        # Minimum Working Days Penalty
        days_used = {assignment["day"] for assignment in assignments}
        missing_days = max(0, course["min_days"] - len(days_used))
        min_days_violation_penalty += 5 * missing_days

        # Room Stability Penalty
        rooms_used = {assignment["room_id"] for assignment in assignments}
        room_stability_penalty += max(0, len(rooms_used) - 1)  # Each extra room adds 1 penalty point

    # Curriculum Compactness Penalty
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
            # Sort by period within the same day
            day_assignments.sort(key=lambda x: x["period"])
            for i in range(len(day_assignments)):
                current = day_assignments[i]
                # Check if the current lecture is isolated
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

    # Calculate total penalty by summing all individual penalties
    total_penalty = (
        room_capacity_penalty +
        min_days_violation_penalty +
        curriculum_compactness_penalty +
        room_stability_penalty
    )

    # Return the total penalty as a tuple
    return (total_penalty,)

toolbox = base.Toolbox()

def updateParticle(data, part, personal_best, local_best, global_best, chi, c1, c2, constraints):
    """
    Updates a particle's velocity and applies it to modify the schedule.
    Uses neighborhood operations for local search.
    """
    # Map rooms to indices
    room_map = {room['id']: i for i, room in enumerate(data['rooms'])}  # Map rooms to indices
    reverse_room_map = {i: room['id'] for i, room in enumerate(data['rooms'])}  # Reverse map to get room names

    for i, entry in enumerate(part):
        r1, r2 = random.random(), random.random()

        # Get the best positions from personal best, local best, and global best
        best_entry = personal_best[i] if personal_best and i < len(personal_best) else entry
        local_best_entry = local_best[i] if local_best and i < len(local_best) else entry
        global_best_entry = global_best[i] if global_best and i < len(global_best) else entry

        # Convert room ids to indices for velocity calculation
        best_room_index = room_map[best_entry["room_id"]]
        local_best_room_index = room_map[local_best_entry["room_id"]]
        global_best_room_index = room_map[global_best_entry["room_id"]]
        current_room_index = room_map[entry["room_id"]]

        # Compute velocity components for day, period, and room (using indices for room)
        velocity_day = chi * (c1 * r1 * (best_entry["day"] - entry["day"]) + c2 * r2 * (local_best_entry["day"] - entry["day"]) + c2 * r2 * (global_best_entry["day"] - entry["day"]))
        velocity_period = chi * (c1 * r1 * (best_entry["period"] - entry["period"]) + c2 * r2 * (local_best_entry["period"] - entry["period"]) + c2 * r2 * (global_best_entry["period"] - entry["period"]))
        velocity_room = chi * (c1 * r1 * (best_room_index - current_room_index) + c2 * r2 * (local_best_room_index - current_room_index) + c2 * r2 * (global_best_room_index - current_room_index))

        # Calculate new positions for day, period, and room index
        new_day = entry["day"] + int(velocity_day)
        new_period = entry["period"] + int(velocity_period)

        # Clamp new_day and new_period within valid bounds
        new_day = max(0, min(data["num_days"] - 1, new_day))
        new_period = max(0, min(data["periods_per_day"] - 1, new_period))

        # Calculate new room index based on the velocity, and map back to room ID
        new_room_index = current_room_index + int(velocity_room)
        new_room_index = max(0, min(len(data["rooms"]) - 1, new_room_index))  # Clamp to valid room index
        new_room = reverse_room_map[new_room_index]  # Convert index back to room ID

        # Save the original state to revert in case of infeasibility
        original_state = (entry["day"], entry["period"], entry["room_id"])

        # Find the course currently assigned to the new position (if any)
        conflicting_entry = next(
            (e for e in part if e["day"] == new_day and e["period"] == new_period and e["room_id"] == new_room),
            None
        )

        # Temporarily assign the new values to the entry
        entry["day"], entry["period"], entry["room_id"] = new_day, new_period, new_room

        if conflicting_entry:
            conflicting_entry["day"], conflicting_entry["period"], conflicting_entry["room_id"] = original_state

        # Check feasibility
        if is_feasible(part, constraints, data["courses"], data["curricula"]):
            continue  # If feasible, no need to revert

        # If not feasible, revert both entries to their original positions
        entry["day"], entry["period"], entry["room_id"] = original_state
        if conflicting_entry:
            conflicting_entry["day"], conflicting_entry["period"], conflicting_entry["room_id"] = new_day, new_period, new_room

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
            return False  # Room conflict, return immediately

        # Assign room to the current course at the given day and period
        room_assignments[(day, period)][room] = course_id

        # Track course assignments for the course ID
        if course_id not in course_assignments:
            course_assignments[course_id] = set()
        
        if (day, period) in course_assignments[course_id]:
            return False  # Course conflict, return immediately
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
                return False  # Curriculum conflict

    # Unavailability Constraints
    for constraint in constraints:
        for entry in schedule:
            if (
                constraint["course"] == entry["course_id"]
                and constraint["day"] == entry["day"]
                and constraint["period"] == entry["period"]
            ):
                return False  # Unavailability conflict

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

def convertQuantum(swarm, rcloud, centre, min_distance, constraints, courses, curricula):
    """
    Reinitializes particles around the swarm's best using Gaussian distribution.
    Ensures minimum distance from previous positions for diversity.
    Keeps solutions feasible (no hard constraint violations).
    """
    for part in swarm:
        for entry, best_entry in zip(part, centre):
            # Initialize original state
            original_day, original_period, original_room = entry["day"], entry["period"], entry["room_id"]

            attempts = 0
            max_attempts = 10  # Limit to avoid infinite loops

            while attempts < max_attempts:
                attempts += 1

                # Generate new positions around the best entry
                new_day = max(0, int(best_entry["day"] + rcloud * random.gauss(0, 1)))
                new_period = max(0, int(best_entry["period"] + rcloud * random.gauss(0, 1)))
                new_room = best_entry["room_id"]

                # Ensure diversity by checking minimum distance
                if abs(new_day - entry["day"]) < min_distance or abs(new_period - entry["period"]) < min_distance:
                    continue

                # Temporarily assign new values
                entry["day"], entry["period"], entry["room_id"] = new_day, new_period, new_room

                # Check feasibility of the updated particle
                if is_feasible(part, constraints, courses, curricula):
                    break  # Feasible solution found, exit the loop
                else:
                    # Revert to original values if not feasible
                    entry["day"], entry["period"], entry["room_id"] = original_day, original_period, original_room

            # If max attempts reached, retain the original values
            if attempts >= max_attempts:
                entry["day"], entry["period"], entry["room_id"] = original_day, original_period, original_room

        # Update fitness after reinitialization
        part.fitness.values = toolbox.evaluate(part)
        part.best = None  # Reset particle's personal best to ensure exploration

def parallel_evaluate(particles, toolbox):
    # Use deep copy to avoid modifying original particles during evaluation
    particles_copy = [copy.deepcopy(part) for part in particles]
    return Parallel(n_jobs=-1)(delayed(toolbox.evaluate)(part) for part in particles_copy)

def main(data, max_iterations=100, verbose=True, no_improvement_limit=10):
    global courses, rooms, curricula
    courses = data["courses"]
    rooms = data["rooms"]
    curricula = data["curricula"]
    constraints = data["constraints"]
    days = data["num_days"]
    periods = data["periods_per_day"]

    toolbox.register("particle", generate, creator.Particle)
    toolbox.register("swarm", tools.initRepeat, creator.Swarm, toolbox.particle)
    toolbox.register(
        "evaluate",
        partial(evaluate_schedule, rooms=rooms, courses=courses, curricula=curricula, constraints=constraints)
    )

    NSWARMS = 1
    NPARTICLES = 5
    NEXCESS = 3
    RCLOUD = 0.5
    BOUNDS = [0, len(rooms) * days * periods]
    min_distance = 2  # Minimum distance threshold for diversity

    population = [toolbox.swarm(n=NPARTICLES) for _ in range(NSWARMS)]

    chi = 0.729  # Constriction coefficient
    c1, c2 = 1, 1.5  # Cognitive and social coefficients

    rexcl = (BOUNDS[1] - BOUNDS[0]) / (2 * NSWARMS**(1.0 / 2))

    # Track the best global fitness
    best_global_fitness = float('inf')
    global_best_particle = None  # Track the global best particle

    for swarm in population:
        swarm.best = None
        swarm.bestfit = creator.FitnessMin((float('inf'),))
        swarm.no_improvement_iters = 0

        # Parallelize fitness evaluation for particles in the swarm (using copies of particles)
        results = parallel_evaluate(swarm, toolbox)

        for i, part in enumerate(swarm):
            part.fitness.values = results[i]
            part.best = toolbox.clone(part)
            part.bestfit = creator.FitnessMin(part.fitness.values)
            if swarm.best is None or part.fitness < swarm.bestfit:
                swarm.best = toolbox.clone(part)
                swarm.bestfit.values = part.fitness.values

        # Update the global best across all swarms
        if global_best_particle is None or swarm.bestfit.values[0] < best_global_fitness:
            best_global_fitness = swarm.bestfit.values[0]
            global_best_particle = toolbox.clone(swarm.best)

    for iteration in range(max_iterations):
        if verbose:
            print(f"Iteration {iteration + 1}/{max_iterations}")

        for i, swarm in enumerate(population):
            for part in swarm:
                if swarm.best:
                    updateParticle(data, part, part.best, swarm.best, global_best_particle, chi, c1, c2, constraints)

                    # Re-evaluate the fitness after updating the particle (using copies of particles)
                    results = parallel_evaluate([part], toolbox)
                    part.fitness.values = results[0]

                    if part.bestfit is None or part.fitness.values < part.bestfit.values:
                        part.best = toolbox.clone(part)
                        part.bestfit.values = part.fitness.values
                        print("Updated part bestfit:", part.bestfit.values)

                    # Print current best in swarm and the current part fitness
                    print("Swarm bestfit before comparison:", swarm.bestfit.values)
                    print("Current particle fitness:", part.fitness.values)

                    if part.fitness.values < swarm.bestfit.values:
                        swarm.best = toolbox.clone(part)
                        swarm.bestfit.values = part.fitness.values
                        swarm.no_improvement_iters = 0
                        print("Updated swarm bestfit with new best particle.")
                    else:
                        swarm.no_improvement_iters += 1
                        print("No improvement in swarm bestfit.")

        # Update global best across all swarms
        best_fitness_in_population = min(swarm.bestfit.values[0] for swarm in population if swarm.bestfit.values)
        if best_fitness_in_population < best_global_fitness:
            best_global_fitness = best_fitness_in_population
            # Update global best particle
            global_best_particle = toolbox.clone([swarm.best for swarm in population if swarm.bestfit.values[0] == best_fitness_in_population][0])

        # Stop if the fitness meets the target of 5 or less
        if best_global_fitness <= 0:
            print(f"\nStopping early as target fitness of 5 or less was reached: {best_global_fitness}")
            break

        # Exclusion mechanism
        if len(population) > 1:
            reinit_swarms = set()
            for s1, s2 in itertools.combinations(range(len(population)), 2):
                if population[s1].best and population[s2].best:
                    distance = math.sqrt(
                        sum(
                            (entry1["day"] - entry2["day"])**2 + (entry1["period"] - entry2["period"])**2
                            for entry1, entry2 in zip(population[s1].best, population[s2].best)
                        )
                    )
                    if distance < rexcl:
                        reinit_swarms.add(s1 if population[s1].bestfit <= population[s2].bestfit else s2)

            for s in reinit_swarms:
                convertQuantum(
                    swarm, 
                    RCLOUD, 
                    swarm.best, 
                    min_distance, 
                    constraints, 
                    data["courses"], 
                    data["curricula"]
                )
                for part in population[s]:
                    part.fitness.values = toolbox.evaluate(part)

        not_converged = sum(
            1 for swarm in population if any(
                math.sqrt(
                    sum(
                        (entry1["day"] - entry2["day"])**2 + (entry1["period"] - entry2["period"])**2
                        for entry1, entry2 in zip(p1, p2)
                    )
                ) > rexcl
                for p1, p2 in itertools.combinations(swarm, 2)
            )
        )

        if not_converged == 0 and len(population) < NSWARMS + NEXCESS:
            new_swarm = toolbox.swarm(n=NPARTICLES)
            new_swarm.best = None
            new_swarm.bestfit = creator.FitnessMin((float('inf'),))
            new_swarm.no_improvement_iters = 0
            for part in new_swarm:
                part.fitness.values = toolbox.evaluate(part)
                part.best = toolbox.clone(part)
                part.bestfit = creator.FitnessMin(part.fitness.values)
                if new_swarm.best is None or part.fitness < new_swarm.bestfit:
                    new_swarm.best = toolbox.clone(part)
                    new_swarm.bestfit.values = part.fitness.values
            population.append(new_swarm)
        elif not_converged > NEXCESS:
            worst_swarm = min(population, key=lambda s: s.bestfit.values[0])
            population.remove(worst_swarm)

    # Determine the best particle across all swarms based on bestfit values if available
    valid_swarms = [swarm for swarm in population if swarm.best is not None and swarm.bestfit is not None]

    if valid_swarms:
        # Find the swarm with the minimum bestfit value
        best_swarm = min(valid_swarms, key=lambda s: s.bestfit.values[0])
        final_best_schedule = [entry for entry in best_swarm.best]
        print("\nFinal Best Solution Found (Fitness):", best_swarm.bestfit.values[0])
    else:
        # Fallback if no swarm has a valid best (highly unlikely with this setup)
        final_best_schedule = None
        print("\nNo solution found.")

    return final_best_schedule

