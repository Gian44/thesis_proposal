import itertools
import math
import random
from deap import base, creator, tools
import operator

# Initialize globals for course and room data, which will be set in main()
courses = []
rooms = []
curricula = []

# Define the fitness and particle classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Particle", list, fitness=creator.FitnessMin, speed=list, best=None, bestfit=None)
creator.create("Swarm", list, best=None, bestfit=None)

# Generate a particle (schedule) based on the required number of lectures for each course
def generate(pclass, courses, rooms, days, periods):
    particle = []
    room_schedule = {}  # Track assigned slots to avoid room conflicts
    course_timeslot_tracker = {}  # Track course assignments to avoid duplicate timeslots for the same course

    for course in courses:
        num_lectures = course.get("num_lectures", 1)
        for _ in range(num_lectures):
            assigned = False
            attempts = 0

            # Try to assign a time slot without conflicts
            while not assigned and attempts < 10:
                room = random.choice(rooms)["id"]
                day = random.randint(0, days - 1)
                period = random.randint(0, periods - 1)

                # Check if the room is free and the course has not been assigned to the same timeslot
                if (room, day, period) not in room_schedule and (course["id"], day, period) not in course_timeslot_tracker:
                    entry = {
                        "course_id": course["id"],
                        "room_id": room,
                        "day": day,
                        "period": period
                    }
                    particle.append(entry)
                    room_schedule[(room, day, period)] = course["id"]
                    course_timeslot_tracker[(course["id"], day, period)] = True  # Track this assignment for the course
                    assigned = True
                attempts += 1

            # Fallback: if assignment couldn't avoid conflicts in 10 tries, assign randomly
            if not assigned:
                room = random.choice(rooms)["id"]
                day = random.randint(0, days - 1)
                period = random.randint(0, periods - 1)
                
                # Track the room and course timeslot regardless of conflicts
                entry = {
                    "course_id": course["id"],
                    "room_id": room,
                    "day": day,
                    "period": period
                }
                particle.append(entry)
                room_schedule[(room, day, period)] = course["id"]
                course_timeslot_tracker[(course["id"], day, period)] = True  # Ensure no duplicate timeslot for the same course

    print("Generated particle:", particle)  # Debug: Check if particle is correctly generated
    return pclass(particle)


# Evaluate a particle (schedule) for constraint violations
def evaluate_schedule(schedule):
    # Define penalties for hard constraints
    room_conflict_penalty = 0
    capacity_violation_penalty = 0
    curriculum_conflict_penalty = 0
    teacher_conflict_penalty = 0

    room_schedule = {}
    curriculum_schedule = {}
    teacher_schedule = {}

    # Calculate hard constraint penalties
    for entry in schedule:
        course_id = entry['course_id']
        room_id = entry['room_id']
        day = entry['day']
        period = entry['period']
        course = next((c for c in courses if c["id"] == course_id), None)
        room = next((r for r in rooms if r["id"] == room_id), None)
        curriculum = next((curr for curr in curricula if course_id in curr["courses"]), None)
        teacher = course["teacher"]

        # Hard constraint: Room conflict check
        if (room_id, day, period) in room_schedule:
            room_conflict_penalty += 5000
        else:
            room_schedule[(room_id, day, period)] = course_id

        # Hard constraint: Capacity check
        if course['num_students'] > room['capacity']:
            capacity_violation_penalty += 5000

        # Curriculum conflict check
        if curriculum:
            if curriculum["id"] not in curriculum_schedule:
                curriculum_schedule[curriculum["id"]] = []
            for (scheduled_day, scheduled_period) in curriculum_schedule[curriculum["id"]]:
                if scheduled_day == day and scheduled_period == period:
                    curriculum_conflict_penalty += 5000
            curriculum_schedule[curriculum["id"]].append((day, period))

        # Teacher conflict check
        if teacher not in teacher_schedule:
            teacher_schedule[teacher] = []
        for (scheduled_day, scheduled_period) in teacher_schedule[teacher]:
            if scheduled_day == day and scheduled_period == period:
                teacher_conflict_penalty += 5000
        teacher_schedule[teacher].append((day, period))

    # Total hard constraint penalty
    hard_constraint_penalty = (
        room_conflict_penalty +
        capacity_violation_penalty +
        curriculum_conflict_penalty +
        teacher_conflict_penalty
    )

    # Define penalties for soft constraints
    room_capacity_utilization_penalty = 0
    min_days_violation_penalty = 0
    curriculum_compactness_penalty = 0
    room_stability_penalty = 0

    # Calculate soft constraint penalties
    for course in courses:
        assigned_periods = [entry for entry in schedule if entry['course_id'] == course["id"]]
        assigned_days = {entry['day'] for entry in assigned_periods}

        for entry in assigned_periods:
            room = next((r for r in rooms if r["id"] == entry['room_id']), None)
            if course['num_students'] > room['capacity']:
                room_capacity_utilization_penalty += (course['num_students'] - room['capacity'])
        if len(assigned_days) < course['min_days']:
            min_days_violation_penalty += 5 * (course['min_days'] - len(assigned_days))
        for day in assigned_days:
            day_periods = sorted(entry['period'] for entry in assigned_periods if entry['day'] == day)
            for i in range(1, len(day_periods)):
                if day_periods[i] != day_periods[i - 1] + 1:
                    curriculum_compactness_penalty += 2
        rooms_used = {entry['room_id'] for entry in assigned_periods}
        if len(rooms_used) > 1:
            room_stability_penalty += (len(rooms_used) - 1)

    # Total soft constraint penalty
    soft_constraint_penalty = (
        room_capacity_utilization_penalty +
        min_days_violation_penalty +
        curriculum_compactness_penalty +
        room_stability_penalty
    )

    # Return combined penalty for both hard and soft constraints
    total_penalty = hard_constraint_penalty + soft_constraint_penalty
    return (total_penalty,)


toolbox = base.Toolbox()

def updateParticle(data, part, best, chi, c, curricula, constraints):
    # Prepare unavailability and curriculum conflict tracking
    unavailability = {course["id"]: set() for course in data["courses"]}
    for constraint in constraints:
        unavailability[constraint["course"]].add((constraint["day"], constraint["period"]))

    used_timeslots = {room["id"]: set() for room in data["rooms"]}
    curriculum_timeslot_tracker = {}
    course_timeslot_tracker = {}  # Track assignments to ensure a course isn't in multiple rooms in the same timeslot

    for entry in part:
        room, day, period = entry["room_id"], entry["day"], entry["period"]
        course_id = entry["course_id"]
        curriculum_id = next((curr["id"] for curr in curricula if course_id in curr["courses"]), None)

        # Update used timeslots and course timeslot tracking
        used_timeslots[room].add((day, period))
        if (course_id, day, period) not in course_timeslot_tracker:
            course_timeslot_tracker[(course_id, day, period)] = room
        if (day, period) not in curriculum_timeslot_tracker:
            curriculum_timeslot_tracker[(day, period)] = set()
        if curriculum_id:
            curriculum_timeslot_tracker[(day, period)].add(curriculum_id)

    for i, entry in enumerate(part):
        if i < len(best):
            best_entry = best[i]
            room = entry["room_id"]
            target_day = int(entry["day"] + chi * (best_entry["day"] - entry["day"]) * c)
            target_period = int(entry["period"] + chi * (best_entry["period"] - entry["period"]) * c)
            course_id = entry["course_id"]
            curriculum_id = next((curr["id"] for curr in curricula if course_id in curr["courses"]), None)

            # Initialize curriculum_timeslot_tracker for the target timeslot if it doesn't exist
            if (target_day, target_period) not in curriculum_timeslot_tracker:
                curriculum_timeslot_tracker[(target_day, target_period)] = set()

            # Check if the target timeslot is unavailable, has curriculum conflicts, or duplicates the course
            if ((target_day, target_period) not in used_timeslots[room] and
                curriculum_id not in curriculum_timeslot_tracker[(target_day, target_period)] and
                (target_day, target_period) not in unavailability[course_id] and
                (course_id, target_day, target_period) not in course_timeslot_tracker):

                # Assign if timeslot is available, conflict-free, and within constraints
                entry["day"], entry["period"] = target_day, target_period
                used_timeslots[room].add((target_day, target_period))
                course_timeslot_tracker[(course_id, target_day, target_period)] = room
                if curriculum_id:
                    curriculum_timeslot_tracker[(target_day, target_period)].add(curriculum_id)
            else:
                # Find an alternative timeslot
                for day in range(data["num_days"]):
                    for period in range(data["periods_per_day"]):
                        if ((day, period) not in used_timeslots[room] and
                            curriculum_id not in curriculum_timeslot_tracker.get((day, period), set()) and
                            (day, period) not in unavailability[course_id] and
                            (course_id, day, period) not in course_timeslot_tracker):

                            # Assign alternative timeslot if found
                            entry["day"], entry["period"] = day, period
                            used_timeslots[room].add((day, period))
                            course_timeslot_tracker[(course_id, day, period)] = room
                            if curriculum_id:
                                curriculum_timeslot_tracker.setdefault((day, period), set()).add(curriculum_id)
                            break
                    else:
                        continue
                    break


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

def main(data, max_iterations=100, verbose=True, no_improvement_limit=10):
    global courses, rooms, curricula
    courses = data["courses"]
    rooms = data["rooms"]
    curricula = data["curricula"]
    constraints = data["constraints"]  # Retrieve the constraints
    days = data["num_days"]
    periods = data["periods_per_day"]

    toolbox.register("particle", generate, creator.Particle, courses=courses, rooms=rooms, days=days, periods=periods)
    toolbox.register("swarm", tools.initRepeat, creator.Swarm, toolbox.particle)
    toolbox.register("evaluate", evaluate_schedule)

    NSWARMS = 1
    NPARTICLES = 7
    NEXCESS = 3
    RCLOUD = 0.5
    BOUNDS = [0, len(rooms) * days * periods]
    min_distance = 2  # Minimum distance threshold for diversity

    population = [toolbox.swarm(n=NPARTICLES) for _ in range(NSWARMS)]

    # Set initial bounds for chi and c
    chi = 0.729 
    c = 1

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
                    # Pass `curricula` to ensure curriculum constraints are checked
                    updateParticle(data, part, swarm.best, chi=chi, c=c, curricula=curricula, constraints=constraints)

                    
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

