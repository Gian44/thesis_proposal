import itertools
import math
import random
from deap import base, creator, tools
from initialize_population import assign_courses
from functools import partial
import operator

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
    Evaluates the fitness of a particle by calculating the penalties
    for soft constraints.

    Args:
        particle (list): The schedule (particle) to evaluate.
        rooms (list): List of rooms with their capacities.
        courses (list): List of courses with their details.
        curricula (list): List of curricula and their courses.
        constraints (list): Hard constraints.

    Returns:
        tuple: A single fitness value combining penalties for hard and soft constraints.
    """
    
    # Initialize penalties
    room_capacity_utilization_penalty = 0
    min_days_violation_penalty = 0
    curriculum_compactness_penalty = 0
    room_stability_penalty = 0

    # Validate rooms
    if not isinstance(rooms, list) or not all(isinstance(r, dict) for r in rooms):
        raise ValueError("Rooms variable must be a list of dictionaries with 'id' and 'capacity' keys.")

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
                room_capacity_utilization_penalty += (course["num_students"] - room_details["capacity"])

        # Minimum Working Days Penalty
        days_used = {assignment["day"] for assignment in assignments}
        if len(days_used) < course["min_days"]:
            min_days_violation_penalty += 5 * (course["min_days"] - len(days_used))

        # Room Stability Penalty
        rooms_used = {assignment["room_id"] for assignment in assignments}
        if len(rooms_used) > 1:  # More than one room used
            room_stability_penalty += (len(rooms_used) - 1)

    # Curriculum Compactness Penalty
    for curriculum in curricula:
        curriculum_courses = curriculum["courses"]
        curriculum_assignments = [
            assignment
            for course_id in curriculum_courses
            for assignment in course_assignments[course_id]
        ]
        curriculum_assignments.sort(key=lambda x: (x["day"], x["period"]))  # Sort by day and period
        for i in range(1, len(curriculum_assignments)):
            current = curriculum_assignments[i]
            previous = curriculum_assignments[i - 1]
            if current["day"] == previous["day"] and current["period"] != previous["period"] + 1:
                curriculum_compactness_penalty += 2  # Non-adjacent lectures in the same day

    # Calculate total penalty
    total_penalty = (
        room_capacity_utilization_penalty +
        min_days_violation_penalty +
        curriculum_compactness_penalty +
        room_stability_penalty 
    )

    return (total_penalty,)

toolbox = base.Toolbox()

def nl_move(data, schedule, constraints, courses, curricula):
    """
    Perform a move operation by randomly reassigning a course to a new valid slot.
    """
    new_schedule = toolbox.clone(schedule)
    for entry in new_schedule:
        # Randomly select a course
        selected_entry = random.choice(new_schedule)
        proposed_day = random.randint(0, data["num_days"] - 1)
        proposed_period = random.randint(0, data["periods_per_day"] - 1)
        proposed_room = random.choice(data["rooms"])["id"]

        # Save original position for revert
        original_day, original_period, original_room = selected_entry["day"], selected_entry["period"], selected_entry["room_id"]

        # Apply the move
        selected_entry["day"], selected_entry["period"], selected_entry["room_id"] = proposed_day, proposed_period, proposed_room

        # Check feasibility
        if is_feasible(new_schedule, constraints, courses, curricula):
            return new_schedule  # Return the new feasible schedule
        else:
            # Revert if infeasible
            selected_entry["day"], selected_entry["period"], selected_entry["room_id"] = original_day, original_period, original_room
    return schedule  # Return original schedule if no valid move is found

def nl_swap(schedule, constraints, courses, curricula):
    """
    Perform a swap operation by exchanging the timeslots of two courses.
    """
    new_schedule = toolbox.clone(schedule)
    entry1, entry2 = random.sample(new_schedule, 2)  # Select two random lectures
    
    # Save the original state for both entries
    entry1_day, entry1_period, entry1_room = entry1["day"], entry1["period"], entry1["room_id"]
    entry2_day, entry2_period, entry2_room = entry2["day"], entry2["period"], entry2["room_id"]

    # Swap their assignments
    entry1["day"], entry1["period"], entry1["room_id"] = entry2_day, entry2_period, entry2_room
    entry2["day"], entry2["period"], entry2["room_id"] = entry1_day, entry1_period, entry1_room

    # Check feasibility
    if is_feasible(new_schedule, constraints, courses, curricula):
        return new_schedule  # Return the new feasible schedule
    else:
        # Revert the swap if infeasible
        entry1["day"], entry1["period"], entry1["room_id"] = entry1_day, entry1_period, entry1_room
        entry2["day"], entry2["period"], entry2["room_id"] = entry2_day, entry2_period, entry2_room
    return schedule  # Return original schedule if no valid swap is found


def updateParticle(data, part, personal_best, global_best, chi, c1, c2, constraints):
    """
    Updates a particle's velocity and applies it to modify the schedule.
    Uses neighborhood operations for local search.
    """
    for i, entry in enumerate(part):
        r1, r2 = random.random(), random.random()
        #print("Entry: " + str(entry))
        # Get the best positions from personal and global bests
        best_entry = personal_best[i] if personal_best and i < len(personal_best) else entry
        #print ("Best Entry: " + str(best_entry))
        global_best_entry = global_best[i] if global_best and i < len(global_best) else entry
        #print ("Global Best Entry: " + str(global_best_entry))
        # Compute velocity components
        velocity_day = chi * (c1 * r1 * (best_entry["day"] - entry["day"]) + c2 * r2 * (global_best_entry["day"] - entry["day"]))
        velocity_period = chi * (c1 * r1 * (best_entry["period"] - entry["period"]) + c2 * r2 * (global_best_entry["period"] - entry["period"]))

        # Calculate new positions
        new_day = max(0, min(data["num_days"] - 1, entry["day"] + int(velocity_day)))
        new_period = max(0, min(data["periods_per_day"] - 1, entry["period"] + int(velocity_period)))
        new_room = random.choice(data["rooms"])["id"]

        #print("New day: " + str(new_day))
        #print("New period: " + str(new_period))
        #print("New room: " + str(new_room))

        # Save original state
        original_day, original_period, original_room = entry["day"], entry["period"], entry["room_id"]

        # Find the course currently assigned to the new position (if any)
        conflicting_entry = next(
            (e for e in part if e["day"] == new_day and e["period"] == new_period and e["room_id"] == new_room),
            None
        )
        #print("Conflicting entry: " + str(conflicting_entry))

        # Temporarily swap positions
        entry["day"], entry["period"], entry["room_id"] = new_day, new_period, new_room
        if conflicting_entry:
            conflicting_entry["day"], conflicting_entry["period"], conflicting_entry["room_id"] = original_day, original_period, original_room

        # Check feasibility
        if is_feasible(part, constraints, data["courses"], data["curricula"]):
            continue
        else:
            # Revert both entries to their original positions
            entry["day"], entry["period"], entry["room_id"] = original_day, original_period, original_room
            if conflicting_entry:
                conflicting_entry["day"], conflicting_entry["period"], conflicting_entry["room_id"] = new_day, new_period, new_room

    # Apply neighborhood operations for further refinement
    part = nl_move(data, part, constraints, data["courses"], data["curricula"])
    part = nl_swap(part, constraints, data["courses"], data["curricula"])

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
    # Room capacity and overlapping lectures
    timetable = {}
    for entry in schedule:
        day = entry["day"]
        period = entry["period"]
        room = entry["room_id"]
        course_id = entry["course_id"]

        if day not in timetable:
            timetable[day] = {}
        if period not in timetable[day]:
            timetable[day][period] = {}
        if room not in timetable[day][period]:
            timetable[day][period][room] = []

        timetable[day][period][room].append(course_id)

        # Room Occupancy Constraint
        if len(timetable[day][period][room]) > 1:
            #print(f"Room conflict: Multiple courses assigned to Room {room} on Day {day}, Period {period}")
            return False

    # Ensure All Lectures are Scheduled and Assigned to Distinct Periods
    for course in courses:
        assignments = [entry for entry in schedule if entry["course_id"] == course["id"]]
        if len(assignments) != course["num_lectures"]:
            #print(f"Lecture count conflict: Course {course['id']} does not have all lectures scheduled.")
            return False
        assigned_periods = {(entry["day"], entry["period"]) for entry in assignments}
        if len(assigned_periods) != len(assignments):
            #print(f"Distinct period conflict: Lectures of Course {course['id']} overlap in periods.")
            return False

    # Teacher Conflict Constraint
    for course in courses:
        assignments = [
            entry for entry in schedule if entry["course_id"] == course["id"]
        ]
        for i, entry1 in enumerate(assignments):
            for entry2 in assignments[i + 1:]:
                if entry1["day"] == entry2["day"] and entry1["period"] == entry2["period"]:
                    #print(f"Teacher conflict: Teacher {teacher} assigned to multiple courses on Day {entry1['day']}, Period {entry1['period']}")
                    return False

    # Curriculum Conflict Constraint
    for curriculum in curricula:
        curriculum_courses = curriculum["courses"]
        curriculum_assignments = [
            entry
            for course_id in curriculum_courses
            for entry in schedule
            if entry["course_id"] == course_id
        ]
        for i, entry1 in enumerate(curriculum_assignments):
            for entry2 in curriculum_assignments[i + 1:]:
                if entry1["day"] == entry2["day"] and entry1["period"] == entry2["period"]:
                    #print(f"Curriculum conflict: Courses {entry1['course_id']} and {entry2['course_id']} overlap on Day {entry1['day']}, Period {entry1['period']}")
                    return False

    # Unavailability Constraints
    for constraint in constraints:
        for entry in schedule:
            if (constraint["course"] == entry["course_id"] and
                constraint["day"] == entry["day"] and
                constraint["period"] == entry["period"]):
                #print(f"Unavailability conflict: Course {entry['course_id']} assigned to unavailable slot (Day {entry['day']}, Period {entry['period']})")
                return False

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

def alternative_search_space(swarm, data, days, periods, all_swarms, swarm_index):
    """
    Reinitializes particles in the swarm to a new search space using the initial solution generation logic.
    Ensures that the new positions do not collide with particles in other swarms.
    
    Args:
        swarm (list): The swarm to reinitialize.
        data (dict): Contains the problem data including rooms, courses, etc.
        days (int): Number of days in the schedule.
        periods (int): Number of periods per day.
        all_swarms (list): List of all swarms in the population to check for collisions.
        swarm_index (int): Index of the current swarm in the population.
    """
    # Helper function to check collisions
    def is_collision(entry, other_swarms):
        for other_swarm_index, other_swarm in enumerate(other_swarms):
            if other_swarm_index == swarm_index:  # Skip checking the current swarm
                continue
            for other_particle in other_swarm:
                for other_entry in other_particle:
                    if (
                        entry["day"] == other_entry["day"]
                        and entry["period"] == other_entry["period"]
                        and entry["room_id"] == other_entry["room_id"]
                    ):
                        return True
        return False

    for part in swarm:
        for entry in part:
            while True:
                # Reinitialize using a similar logic to initial particle generation
                new_day = random.randint(0, days - 1)
                new_period = random.randint(0, periods - 1)
                new_room = random.choice(data["rooms"])["id"]

                # Assign the new values temporarily
                entry["day"], entry["period"], entry["room_id"] = new_day, new_period, new_room

                # Check for collisions with other swarms
                if not is_collision(entry, all_swarms):
                    break  # Exit the loop if no collision is detected

        # Re-evaluate fitness after reinitialization
        part.fitness.values = toolbox.evaluate(part)
        part.best = None  # Reset personal best to focus on new space

def main(data, max_iterations=50, verbose=True, no_improvement_limit=10):
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
    c1, c2 = 1, 1  # Cognitive and social coefficients

    rexcl = (BOUNDS[1] - BOUNDS[0]) / (2 * NSWARMS**(1.0 / 2))

    # Track the best global fitness
    best_global_fitness = float('inf')

    for swarm in population:
        swarm.best = None
        swarm.bestfit = creator.FitnessMin((float('inf'),))
        swarm.no_improvement_iters = 0

        for part in swarm:
            part.fitness.values = toolbox.evaluate(part)
            part.best = toolbox.clone(part)
            part.bestfit = creator.FitnessMin(part.fitness.values)
            if swarm.best is None or part.fitness < swarm.bestfit:
                swarm.best = toolbox.clone(part)
                swarm.bestfit.values = part.fitness.values

    for iteration in range(max_iterations):
        if verbose:
            print(f"Iteration {iteration + 1}/{max_iterations}")

        for i, swarm in enumerate(population):
            for part in swarm:
                if swarm.best:
                    updateParticle(data, part, part.best, swarm.best, chi, c1, c2, constraints)
                    
                    # Re-evaluate the fitness after updating the particle
                    part.fitness.values = toolbox.evaluate(part)
                    print(f"Part fitness: {part.fitness.values}")
                    
                    # Update part.best and part.bestfit if current fitness is better
                    if part.bestfit is None or part.fitness.values < part.bestfit.values:
                        part.best = toolbox.clone(part)
                        part.bestfit.values = part.fitness.values  # Ensure part.bestfit is updated correctly
                        print("Updated part bestfit:", part.bestfit.values)

                    # Print current best in swarm and the current part fitness
                    print("Swarm bestfit before comparison:", swarm.bestfit.values)
                    print("Current particle fitness:", part.fitness.values)

                    # Compare using the .values attribute explicitly
                    if part.fitness.values < swarm.bestfit.values:
                        swarm.best = toolbox.clone(part)
                        swarm.bestfit.values = part.fitness.values  # Update with new best fitness
                        swarm.no_improvement_iters = 0
                        print("Updated swarm bestfit with new best particle.")
                    else:
                        swarm.no_improvement_iters += 1
                        print("No improvement in swarm bestfit.")

        # Track the best global fitness
        best_fitness_in_population = min(swarm.bestfit.values[0] for swarm in population if swarm.bestfit.values)
        if best_fitness_in_population < best_global_fitness:
            best_global_fitness = best_fitness_in_population

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

    if final_best_schedule:
        print("\nFinal Best Schedule Analysis:")
        soft_constraints = analyze_soft_constraints(
            final_best_schedule,
            data["courses"],
            data["rooms"],
            data["curricula"]
        )

        print("\nSoft Constraint Violations:")
        for constraint, penalty in soft_constraints.items():
            print(f"{constraint}: {penalty}")

        print("\nTotal Soft Constraint Penalty:", sum(soft_constraints.values()))
    else:
        print("No feasible solution found.")


    return final_best_schedule


def analyze_soft_constraints(schedule, courses, rooms, curricula):
    """
    Analyze and calculate soft constraint penalties in the given schedule.

    Args:
        schedule (list): The final best schedule.
        courses (list): List of all courses with metadata.
        rooms (list): List of all rooms with metadata.
        curricula (list): List of curricula and their courses.

    Returns:
        dict: A dictionary containing penalties for each soft constraint.
    """
    # Initialize penalties
    room_capacity_penalty = 0
    min_days_violation_penalty = 0
    curriculum_compactness_penalty = 0
    room_stability_penalty = 0

    # Convert schedule into a timetable-like structure
    timetable = {}
    for entry in schedule:
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

    # Return the penalties
    return {
        "room_capacity_penalty": room_capacity_penalty,
        "min_days_violation_penalty": min_days_violation_penalty,
        "curriculum_compactness_penalty": curriculum_compactness_penalty,
        "room_stability_penalty": room_stability_penalty,
    }

if __name__ == "__main__":
    # Example usage:
    # Assuming `data` is a dictionary that contains 'courses', 'rooms', 'curricula', etc.
    data = {
        "courses": [
            # example course data here...
        ],
        "rooms": [
            # example room data here...
        ],
        "curricula": [
            # example curriculum data here...
        ],
        "num_days": 5,
        "periods_per_day": 6
    }

    max_iterations = 100
    best_schedule = main(data, max_iterations=max_iterations)

