import itertools
import math
import random
from deap import base, creator, tools
import operator
from initialize_population import assign_courses

# Initialize globals for course and room data, which will be set in main()
courses = []
rooms = []
curricula = []

# Define the fitness and particle classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Particle", list, fitness=creator.FitnessMin, speed=list, best=None, bestfit=None)
creator.create("Swarm", list, best=None, bestfit=None)

# Generate a particle (schedule) using Graph Based
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



def evaluate_schedule(particle):
    """
    Evaluates a particle's schedule for soft constraint penalties.
    Args:
        particle: A list of schedule entries representing a particle.
    Returns:
        A tuple containing the total penalty score.
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

def updateParticle(data, part, personal_best, global_best, chi, c1, c2, constraints):
    """
    Updates a particle's position based on its velocity and resolves conflicts using swap-based heuristic.
    """
    for i, entry in enumerate(part):
        r1, r2 = random.random(), random.random()

        # Fallback to the current entry if personal_best or global_best is None
        best_entry = personal_best[i] if personal_best and i < len(personal_best) else entry
        global_best_entry = global_best[i] if global_best and i < len(global_best) else entry

        # Calculate new values based on velocity equation
        new_day = int(entry["day"] + chi * (
            c1 * r1 * (best_entry["day"] - entry["day"]) +
            c2 * r2 * (global_best_entry["day"] - entry["day"])
        ))
        new_period = int(entry["period"] + chi * (
            c1 * r1 * (best_entry["period"] - entry["period"]) +
            c2 * r2 * (global_best_entry["period"] - entry["period"])
        ))
        new_room = random.choice(data["rooms"])["id"]

        # Keep new values within bounds
        new_day = max(0, min(new_day, data["num_days"] - 1))
        new_period = max(0, min(new_period, data["periods_per_day"] - 1))

        # Feasibility Check
        if is_feasible(part, new_day, new_period, new_room, entry["course_id"], constraints):
            # Update position if feasible
            entry["day"], entry["period"], entry["room_id"] = new_day, new_period, new_room
            print("Position updated")
        else:
            # Use the swap-based heuristic to resolve the conflict
            if not swap_events(part, constraints):
                print("Conflict detected, but no feasible swaps found. The schedule may remain infeasible.")


def is_feasible(part, day, period, room, course_id, constraints):
    for entry in part:
        if entry["room_id"] == room and entry["day"] == day and entry["period"] == period:
            return False  # Room conflict

        if entry["course_id"] == course_id and entry["day"] == day and entry["period"] == period:
            return False  # Duplicate timeslot for the same course

    for constraint in constraints:
        if constraint["course"] == course_id and (day, period):
            return False  # Violates unavailability constraint

    return True

def swap_events(part, constraints):
    """
    Attempts to resolve conflicts in the schedule by swapping two events.
    """
    for i, entry1 in enumerate(part):
        for j, entry2 in enumerate(part):
            if i != j:  # Ensure we're not swapping an event with itself
                # Temporarily swap the entries
                temp_day, temp_period, temp_room = entry1["day"], entry1["period"], entry1["room_id"]
                entry1["day"], entry1["period"], entry1["room_id"] = entry2["day"], entry2["period"], entry2["room_id"]
                entry2["day"], entry2["period"], entry2["room_id"] = temp_day, temp_period, temp_room

                # Check feasibility after the swap
                if is_feasible(part, entry1["day"], entry1["period"], entry1["room_id"], entry1["course_id"], constraints) and \
                   is_feasible(part, entry2["day"], entry2["period"], entry2["room_id"], entry2["course_id"], constraints):
                    return True  # Successful swap, keep changes
                else:
                    # Revert the swap if it creates conflicts
                    entry1["day"], entry1["period"], entry1["room_id"] = temp_day, temp_period, temp_room
                    entry2["day"], entry2["period"], entry2["room_id"] = entry2["day"], entry2["period"], entry2["room_id"]

    return False  # No successful swap found

def convertQuantum(swarm, rcloud, centre, min_distance=2):
    """Reinitializes particles around the swarm's best using Gaussian distribution.
       Ensures minimum distance from previous positions for diversity."""
    for part in swarm:
        for entry, best_entry in zip(part, centre):
            new_day = max(0, int(best_entry["day"] + rcloud * random.gauss(0, 1)))
            new_period = max(0, int(best_entry["period"] + rcloud * random.gauss(0, 1)))
            
            # Ensure diversity by checking if new position is distinct enough from previous position
            if abs(new_day - entry["day"]) < min_distance and abs(new_period - entry["period"]) < min_distance:
                # Reapply Gaussian offset if too close to previous position
                new_day += int(min_distance * random.choice([-1, 1]))
                new_period += int(min_distance * random.choice([-1, 1]))

            # Assign new position
            entry["day"] = max(0, min(new_day, centre[0]["day"]))
            entry["period"] = max(0, min(new_period, centre[0]["period"]))
        
        # Update fitness after reinitialization
        part.fitness.values = toolbox.evaluate(part)
        part.best = None  # Reset particle's personal best to ensure it explores

def alternative_search_space(swarm, data, days, periods):
    """Randomly reinitializes particles to a new search space if current space is unfeasible."""
    for part in swarm:
        for entry in part:
            # Assign random positions to encourage exploration of a new search space
            entry["day"] = random.randint(0, days - 1)
            entry["period"] = random.randint(0, periods - 1)
            entry["room_id"] = random.choice(data["rooms"])["id"]

        # Re-evaluate fitness after alternative search space initialization
        part.fitness.values = toolbox.evaluate(part)
        part.best = None  # Reset personal best to focus on new space

def main(data, max_iterations=5, verbose=True, no_improvement_limit=10):
    global courses, rooms, curricula
    courses = data["courses"]
    rooms = data["rooms"]
    curricula = data["curricula"]
    constraints = data["constraints"]  # Retrieve the constraints
    days = data["num_days"]
    periods = data["periods_per_day"]

    toolbox.register("particle", generate, creator.Particle)
    toolbox.register("swarm", tools.initRepeat, creator.Swarm, toolbox.particle)
    toolbox.register("evaluate", evaluate_schedule)

    NSWARMS = 1
    NPARTICLES = 7
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

            if verbose and swarm.bestfit is not None:
                print(f"Swarm {i+1} Best Fitness: {swarm.bestfit.values[0]}")

            if swarm.no_improvement_iters > no_improvement_limit:
                convertQuantum(swarm, RCLOUD, swarm.best, min_distance)
                if swarm.no_improvement_iters > no_improvement_limit * 2:
                    alternative_search_space(swarm, data, days, periods)
                swarm.no_improvement_iters = 0

        # Track the best global fitness
        best_fitness_in_population = min(swarm.bestfit.values[0] for swarm in population if swarm.bestfit.values)
        if best_fitness_in_population < best_global_fitness:
            best_global_fitness = best_fitness_in_population

        # Stop if the fitness meets the target of 5 or less
        if best_global_fitness <= 5:
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
                convertQuantum(population[s], RCLOUD, population[s].best, min_distance)
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

