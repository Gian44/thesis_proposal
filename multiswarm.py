import itertools
import math
import random
from deap import base, creator, tools
from initialize_population2 import assign_courses
from functools import partial

# Initialize globals for course and room data, which will be set in main()
courses = []
rooms = []
curricula = []
data = []

# Define the fitness and particle classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Particle", list, fitness=creator.FitnessMin, speed=list, best=None, bestfit=None, is_quantum=False)
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

def calculate_min_days_violation_penalty(particle, courses):
    min_days_violation_penalty = 0
    # Track course assignments
    course_assignments = {course["id"]: [] for course in courses}
    for entry in particle:
        if entry["course_id"] != -1:
            course_assignments[entry["course_id"]].append({"day": entry["day"]})

    # Calculate penalty for each course
    for course in courses:
        course_id = course["id"]
        assignments = course_assignments[course_id]
        days_used = {assignment["day"] for assignment in assignments}
        missing_days = max(0, course["min_days"] - len(days_used))
        min_days_violation_penalty += 5 * missing_days
    return min_days_violation_penalty

def calculate_room_stability_penalty(particle, courses):
    room_stability_penalty = 0
    # Track course assignments
    course_assignments = {course["id"]: [] for course in courses}
    for entry in particle:
        if entry["course_id"] != -1:
            course_assignments[entry["course_id"]].append({"room_id": entry["room_id"]})

    # Calculate penalty for each course
    for course in courses:
        course_id = course["id"]
        assignments = course_assignments[course_id]
        rooms_used = {assignment["room_id"] for assignment in assignments}
        room_stability_penalty += max(0, len(rooms_used) - 1)  # Each extra room adds 1 penalty point
    return room_stability_penalty

def calculate_curriculum_compactness_penalty(particle, curricula, courses):
    curriculum_compactness_penalty = 0
    # Track course assignments
    course_assignments = {course["id"]: [] for course in courses}
    for entry in particle:
        if entry["course_id"] != -1:
            course_assignments[entry["course_id"]].append({"day": entry["day"], "period": entry["period"]})

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
    room_capacity_penalty = calculate_room_capacity_penalty(particle, rooms, courses)
    min_days_violation_penalty = calculate_min_days_violation_penalty(particle, courses)
    room_stability_penalty = calculate_room_stability_penalty(particle, courses)
    curriculum_compactness_penalty = calculate_curriculum_compactness_penalty(particle, curricula, courses)

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

def updateParticle(data, part, personal_best, local_best, chi, c1, c2, constraints):
    """
    Updates a randomly selected particle's velocity and applies it to modify the schedule.
    Uses neighborhood operations for local search (personal best and local best).
    Ensures the move is beneficial by checking penalties before and after.
    """

    r1, r2 = random.random(), random.random()

    # Map rooms to indices
    room_map = {room['id']: i for i, room in enumerate(data['rooms'])}
    reverse_room_map = {i: room['id'] for i, room in enumerate(data['rooms'])}

    # Randomly select a course to update
    selected_course_index = random.randint(0, len(part) - 1)
    entry = part[selected_course_index]

    # Save the original penalty
    original_penalty = evaluate_schedule(part, data["rooms"], data["courses"], data["curricula"], constraints)

    # Get the best positions from personal best, and local best
    best_entry = personal_best[selected_course_index] if personal_best and selected_course_index < len(personal_best) else entry
    local_best_entry = local_best[selected_course_index] if local_best and selected_course_index < len(local_best) else entry

    # Convert room ids to indices for velocity calculation
    best_room_index = room_map[best_entry["room_id"]]
    local_best_room_index = room_map[local_best_entry["room_id"]]
    current_room_index = room_map[entry["room_id"]]

    # Compute velocity components for day, period, and room
    velocity_day = chi * (c1 * r1 * (best_entry["day"] - entry["day"]) + c2 * r2 * (local_best_entry["day"] - entry["day"]))
    velocity_period = chi * (c1 * r1 * (best_entry["period"] - entry["period"]) + c2 * r2 * (local_best_entry["period"] - entry["period"]))
    velocity_room = chi * (c1 * r1 * (best_room_index - current_room_index) + c2 * r2 * (local_best_room_index - current_room_index))

    # Apply custom rounding to the velocities
    new_day = entry["day"] + round(velocity_day)
    new_period = entry["period"] + round(velocity_period)

    min_day = min(data["num_days"] - 1, new_day)
    min_period = min(data["periods_per_day"] - 1, new_period)

    # Clamp new_day and new_period within valid bounds
    new_day = max(0, min_day)
    new_period = max(0, min_period)

    # Calculate new room index based on the velocity, and map back to room ID
    new_room_index = current_room_index + round(velocity_room)
    new_room_index = max(0, min(len(data["rooms"]) - 1, new_room_index))  # Clamp to valid room index
    new_room = reverse_room_map[new_room_index]  # Convert index back to room ID

    # Save the original state to revert in case of infeasibility
    original_state = (entry["day"], entry["period"], entry["room_id"])

    # Temporarily assign the new values to the entry
    entry["day"], entry["period"], entry["room_id"] = new_day, new_period, new_room

    # Recalculate penalty after the move
    new_penalty = evaluate_schedule(part, data["rooms"], data["courses"], data["curricula"], constraints)

    # Check feasibility and penalty improvement
    if not is_feasible(part, constraints, data["courses"], data["curricula"]) or new_penalty >= original_penalty:
        # Revert changes if not feasible or penalty did not improve
        entry["day"], entry["period"], entry["room_id"] = original_state
        print(f"Update Particle: Move reverted - Original Penalty: {original_penalty}, New Penalty: {new_penalty}")
    else:
        print(f"Update Particle: Move accepted - Penalty improved from {original_penalty} to {new_penalty}")

def time_move(particle, constraints, rooms, courses, days, periods):
    """
    Implements the time move:
    1. Randomly selects a lecture and a new time slot (day, period).
    2. Verifies feasibility and evaluates penalty before deciding to accept or revert the move.
    """

    # Step 1: Randomly select a lecture
    entry = random.choice(particle)
    original_day, original_period = entry["day"], entry["period"]

    print(f"Time Move: Selected lecture at Day {original_day}, Period {original_period}")

    # Calculate the original penalty before the move
    original_penalty = evaluate_schedule(particle, rooms, courses, curricula, constraints)

    # Step 2: Try new time slots starting from a random day and period
    random_day_offset = random.randint(0, days - 1)
    random_period_offset = random.randint(0, periods - 1)

    for d in range(days):
        for p in range(periods):
            new_day = (d + random_day_offset) % days
            new_period = (p + random_period_offset) % periods

            # Skip the same time slot as the original
            if new_day == original_day and new_period == original_period:
                continue

            # Temporarily move the lecture to the new time slot
            entry["day"], entry["period"] = new_day, new_period

            # Step 3: Check feasibility
            if is_feasible(particle, constraints, courses, curricula):
                # Step 4: Evaluate the new penalty
                new_penalty = evaluate_schedule(particle, rooms, courses, curricula, constraints)

                # Accept the move if it reduces the penalty
                if new_penalty < original_penalty:
                    print(f"Time Move: Successful! Penalty improved from {original_penalty} to {new_penalty}.")
                    return  # Keep the move
                else:
                    print(f"Time Move: No improvement. Penalty {new_penalty} >= {original_penalty}. Reverting...")

            # Revert the move if infeasible or no improvement
            entry["day"], entry["period"] = original_day, original_period

    # Step 5: Revert if no valid move was found
    print(f"Time Move: Reverted to original Day {original_day}, Period {original_period}. No valid improvement found.")

def room_move(particle, constraints, rooms, courses, days, periods):
    """
    Implements the room move:
    1. Randomly selects a lecture and a target room.
    2. Checks for conflicts in the target room.
    3. Verifies feasibility and evaluates penalty before deciding to accept or revert the move.
    """
    # Select a random lecture to move
    entry = random.choice(particle)
    original_room = entry["room_id"]
    day, period = entry["day"], entry["period"]

    print(f"Room Move: Selected lecture in Room {original_room}, Day {day}, Period {period}")

    # Calculate the original penalty before moving
    original_penalty = evaluate_schedule(particle, rooms, courses, curricula, constraints)

    # Randomly shuffle rooms and try them sequentially
    room_offset = random.randint(0, len(rooms) - 1)
    for i in range(len(rooms)):
        room_index = (i + room_offset) % len(rooms)
        selected_room = rooms[room_index]

        # Skip if the selected room is the same as the current room
        if selected_room["id"] == original_room:
            continue

        # Temporarily move the lecture to the new room
        entry["room_id"] = selected_room["id"]

        # Check for feasibility
        if is_feasible(particle, constraints, courses, curricula):
            # Calculate the new penalty
            new_penalty = evaluate_schedule(particle, rooms, courses, curricula, constraints)

            # Accept the move if it improves the penalty
            if new_penalty < original_penalty:
                print(f"Room Move: Successful! Penalty improved from {original_penalty} to {new_penalty}.")
                return  # Keep the move
            else:
                print(f"Room Move: No improvement. Penalty {new_penalty} >= {original_penalty}. Reverting...")
        
        # Revert the move if infeasible or no improvement
        entry["room_id"] = original_room

    print(f"Room Move: Reverted to original Room {original_room}. No valid improvement found.")

def room_stability_move(particle, constraints, courses, rooms):
    """
    Implements room stability move closely following the logic of the provided move_2 function.
    1. Randomly selects a course and moves all its lectures to a new room.
    2. Resolves conflicts by reassigning conflicting lectures to the original room.
    3. Checks for feasibility and evaluates soft constraints to decide whether to keep the move.
    """

    # Step 1: Map room IDs for lookup
    room_map = {room["id"]: room for room in rooms}

    # Step 2: Randomly select a course
    course_ids = list(set(lecture["course_id"] for lecture in particle))
    if not course_ids:
        return particle  # No courses to move

    selected_course_id = random.choice(course_ids)
    course_lectures = [lecture for lecture in particle if lecture["course_id"] == selected_course_id]

    # Step 3: Randomly select a new room
    all_room_ids = list(room_map.keys())
    new_room_id = random.choice(all_room_ids)

    # Step 4: Track original solution state and conflicts
    new_solution = [lecture.copy() for lecture in particle]  # Create a deep copy of the solution
    changes = []  # Track lectures moved
    conflicts = []  # Track conflicts resolved

    # Step 5: Attempt to reassign all lectures of the course to the new room
    for lecture in course_lectures:
        day, period, current_room_id = lecture["day"], lecture["period"], lecture["room_id"]

        # Check for conflicts in the new room at the same day-period
        conflicting_lectures = [
            l for l in new_solution
            if l["room_id"] == new_room_id and l["day"] == day and l["period"] == period
        ]

        # Handle conflicting lectures
        for conflict in conflicting_lectures:
            conflicts.append((conflict, conflict["room_id"]))  # Backup conflict's state
            conflict["room_id"] = current_room_id  # Move conflict to the old room

        # Update the lecture to the new room
        changes.append((lecture, lecture["room_id"]))  # Backup original state
        lecture["room_id"] = new_room_id

    # Step 6: Check feasibility
    if not is_feasible(new_solution, constraints, courses, curricula):
        # Revert all changes and conflicts if infeasible
        for lecture, original_room in changes:
            lecture["room_id"] = original_room
        for conflict, original_room in conflicts:
            conflict["room_id"] = original_room
        print("Room Stability Move: Reverted due to infeasibility.")
        return particle  # Return original solution

    # Step 7: Evaluate soft constraints
    original_penalty = evaluate_schedule(particle, rooms, courses, curricula, constraints)
    new_penalty = evaluate_schedule(particle, rooms, courses, curricula, constraints)

    # Step 8: Accept or reject the move
    if new_penalty < original_penalty:
        print(f"Room Stability Move: Successful - Penalty improved from {original_penalty} to {new_penalty}")
        return new_solution  # Accept the improved solution
    else:
        # Revert changes if no improvement
        for lecture, original_room in changes:
            lecture["room_id"] = original_room
        for conflict, original_room in conflicts:
            conflict["room_id"] = original_room
        print("Room Stability Move: Reverted - No improvement.")
        return particle  # Return original solution

def calculate_individual_room_stability_penalty(lectures, room_id=None):
    """
    Calculate the room stability penalty for a given set of lectures.
    If `room_id` is provided, assumes lectures are assigned to this room for penalty calculation.
    """
    if room_id is None:
        room_id = lectures[0]["room_id"]  # Assume all lectures are in the same room

    unique_rooms_used = len(set(lecture["room_id"] for lecture in lectures))
    return max(0, unique_rooms_used - 1)  # Penalty is the number of additional rooms used

def min_working_days_move(particle, constraints, courses, rooms, num_days):
    violating_courses = [
        c for c in courses if len(set(e["day"] for e in particle if e["course_id"] == c["id"])) < c["min_days"]
    ]
    if not violating_courses:
        print("Min Working Days Move: No violating courses found.")
        return

    course = random.choice(violating_courses)
    course_id = course["id"]
    print(f"Min Working Days Move: Selected Course {course_id} with violations.")

    original_penalty = calculate_min_days_violation_penalty(particle, courses)
    course_lectures = [e for e in particle if e["course_id"] == course_id]
    days_used = set(lecture["day"] for lecture in course_lectures)
    overcrowded_days = {day for day in days_used if sum(lecture["day"] == day for lecture in course_lectures) > 1}

    for lecture in course_lectures:
        if lecture["day"] not in overcrowded_days:
            continue

        original_day, original_room = lecture["day"], lecture["room_id"]

        for day in range(num_days):
            if day in days_used:
                continue

            for room in rooms:
                if room["capacity"] < course["num_students"]:
                    continue

                lecture["day"], lecture["room_id"] = day, room["id"]

                if is_feasible(particle, constraints, courses, curricula):
                    new_penalty = calculate_min_days_violation_penalty(particle, courses)
                    if new_penalty < original_penalty:
                        print(f"Min Working Days Move: Successful move for Course {course_id}.")
                        return

                lecture["day"], lecture["room_id"] = original_day, original_room

    print(f"Min Working Days Move: Reverted - No valid moves for Course {course_id}")


def curriculum_compactness_move(particle, constraints, curricula, rooms, num_days, num_periods, courses):
    # Step 1: Randomly select a curriculum
    curriculum = random.choice(curricula)
    curriculum_courses = curriculum["courses"]
    curriculum_lectures = [e for e in particle if e["course_id"] in curriculum_courses]

    # Calculate the number of students for the curriculum by summing its courses' students
    curriculum_num_students = sum(
        course["num_students"] for course in courses if course["id"] in curriculum_courses
    )

    # Step 2: Identify lectures that are not adjacent to any other lectures
    non_adjacent_lectures = []
    for lecture in curriculum_lectures:
        day, period = lecture["day"], lecture["period"]
        is_adjacent = False

        for other_lecture in curriculum_lectures:
            if other_lecture == lecture:
                continue
            if other_lecture["day"] == day and abs(other_lecture["period"] - period) == 1:
                is_adjacent = True
                break

        if not is_adjacent:
            non_adjacent_lectures.append(lecture)

    if not non_adjacent_lectures:
        return  # No non-adjacent lectures, exit early

    # Step 3: Randomly select a non-adjacent lecture
    selected_lecture = random.choice(non_adjacent_lectures)
    original_day, original_period, original_room = (
        selected_lecture["day"],
        selected_lecture["period"],
        selected_lecture["room_id"],
    )

    # Step 4: Try to find the best adjacent slot and room
    random_day_offset = random.randint(0, num_days - 1)
    random_period_offset = random.randint(0, num_periods - 1)
    random_room_offset = random.randint(0, len(rooms) - 1)

    for d in range(num_days):
        for p in range(num_periods):
            new_day = (d + random_day_offset) % num_days
            new_period = (p + random_period_offset) % num_periods

            # Check for adjacent lectures
            has_adjacent = any(
                other_lecture["day"] == new_day and abs(other_lecture["period"] - new_period) == 1
                for other_lecture in curriculum_lectures
                if other_lecture != selected_lecture
            )
            if not has_adjacent:
                continue

            for r in range(len(rooms)):
                room = rooms[(r + random_room_offset) % len(rooms)]
                if room["capacity"] < curriculum_num_students:
                    continue

                selected_lecture["day"], selected_lecture["period"], selected_lecture["room_id"] = new_day, new_period, room["id"]

                if is_feasible(particle, constraints, courses, curricula):
                    print(f"Curriculum Compactness Move: Moved lecture to day {new_day}, period {new_period}, room {room['id']}")
                    return  # Successful move

    # Step 6: Revert changes if no valid move is found
    selected_lecture["day"], selected_lecture["period"], selected_lecture["room_id"] = (
        original_day,
        original_period,
        original_room,
    )
    print(f"Curriculum Compactness Move: No valid move found for curriculum {curriculum['id']}")

# Apply neighborhood moves
def apply_neighborhood_moves(particle, data, constraints, days, periods):
    moves = [
        (time_move, 1),
        (room_move, 1),
        (room_stability_move, 1),
        (min_working_days_move, 1),
        (curriculum_compactness_move, 0.1),
    ]
    
    # Create weighted choices
    weighted_moves = [(move, weight) for move, weight in moves for _ in range(int(weight * 10))]
    selected_move = random.choice(weighted_moves)[0]  # Randomly select based on weights
    
    # Execute the selected move
    if selected_move == curriculum_compactness_move:
        selected_move(particle, constraints, data["curricula"], data["rooms"], days, periods, courses)
    elif selected_move == min_working_days_move:
        selected_move(particle, constraints, data["courses"], data["rooms"], days)
    elif selected_move == room_stability_move:
        selected_move(particle, constraints, data["courses"], data["rooms"])
    else:
        selected_move(particle, constraints, data["rooms"], data["courses"], days, periods)

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

def convertQuantum(swarm, rcloud, centre, min_distance, constraints, courses, curricula, rooms):
    """
    Converts all particles in the swarm to quantum particles.
    Reinitializes each particle's position around the swarm's global best (centre) using Gaussian distribution.
    Ensures a minimum distance from previous positions and checks feasibility.
    """
    # Map rooms to indices for efficient access
    room_map = {room['id']: i for i, room in enumerate(rooms)}  # Map rooms to indices
    reverse_room_map = {i: room['id'] for i, room in enumerate(rooms)}  # Reverse map to get room names

    for part in swarm:
        original_state = [entry.copy() for entry in part]  # Store original states to revert in case of infeasibility
        attempts = 0
        max_attempts = 10  # Limit to avoid infinite loops

        while attempts < max_attempts:
            attempts += 1

            # Generate new positions around the best-found position (centre) for each course assignment
            for entry, best_entry in zip(part, centre):
                best_room_index = room_map[best_entry["room_id"]]

                new_day = max(0, int(best_entry["day"] + rcloud * random.gauss(0, 1)))  # Random movement in day
                new_period = max(0, int(best_entry["period"] + rcloud * random.gauss(0, 1)))  # Random movement in period

                # Allow room to be changed, use the best room or sample a new one randomly
                new_room_index = int(best_room_index + rcloud * random.gauss(0, 1))  # Randomize new room
                new_room_index = max(0, min(len(rooms) - 1, new_room_index))  # Clamp within valid room indices
                new_room = reverse_room_map[new_room_index]

                # Ensure diversity by checking minimum distance
                if abs(new_day - entry["day"]) < min_distance or abs(new_period - entry["period"]) < min_distance:
                    continue  # Skip if new position is too close to the original

                # Temporarily assign new values to this particle's entry
                entry["day"], entry["period"], entry["room_id"] = new_day, new_period, new_room

            # Check feasibility of the updated particle (all assignments)
            if is_feasible(part, constraints, courses, curricula):
                break  # If feasible, exit the loop

            # If not feasible, revert the particle to its original state
            for entry, original_entry in zip(part, original_state):
                entry["day"], entry["period"], entry["room_id"] = original_entry["day"], original_entry["period"], original_entry["room_id"]

        # If max attempts reached, retain the original values (no feasible solution found)
        if attempts >= max_attempts:
            for entry, original_entry in zip(part, original_state):
                entry["day"], entry["period"], entry["room_id"] = original_entry["day"], original_entry["period"], original_entry["room_id"]

        # Update fitness after reinitialization
        part.fitness.values = toolbox.evaluate(part)
        part.best = None  # Reset particle's personal best to allow exploration

# Main loop to simulate the Multi-Swarm Particle Swarm Optimization
def main(data, max_iterations=2000, verbose=True):
    global courses, rooms, curricula
    courses = data["courses"]
    rooms = data["rooms"]
    curricula = data["curricula"]
    constraints = data["constraints"]
    days = data["num_days"]
    periods = data["periods_per_day"]
    lectures = 0

    for course in courses:
        lectures += course["num_lectures"]

    toolbox.register("particle", generate, creator.Particle)
    toolbox.register("swarm", tools.initRepeat, creator.Swarm, toolbox.particle)
    toolbox.register(
        "evaluate",
        partial(evaluate_schedule, rooms=rooms, courses=courses, curricula=curricula, constraints=constraints)
    )

    NSWARMS = 1
    NPARTICLES = 5
    NEXCESS = 3
    RCLOUD = 0.3
    NDIM = 3
    BOUNDS = [0, len(rooms) * days * periods]
    min_distance = 3  # Minimum distance threshold for diversity

    population = [toolbox.swarm(n=NPARTICLES) for _ in range(NSWARMS)]


    chi, c1, c2 = 0.729, 1.49445, 1.49445

    # Track the best global fitness
    best_global_fitness = float('inf')
    global_best_particle = None  # Track the global best particle
    last_global_best_update = -1  # This variable tracks the iteration number

    for swarm in population:
        swarm.best = None
        swarm.bestfit = creator.FitnessMin((float('inf'),))
        swarm.no_improvement_iters = 0

        # Evaluate fitness for all particles in the swarm 
        for part in swarm:
            part.fitness.values = toolbox.evaluate(part)
            part.best = toolbox.clone(part)
            part.bestfit = creator.FitnessMin(part.fitness.values)
            if swarm.best is None or part.fitness < swarm.bestfit:
                swarm.best = toolbox.clone(part)
                swarm.bestfit.values = part.fitness.values

        # Update the global best across all swarms
        if global_best_particle is None or swarm.bestfit.values[0] < best_global_fitness:
            best_global_fitness = swarm.bestfit.values[0]
            global_best_particle = toolbox.clone(swarm.best)
    # Track initial fitness values here
    initial_fitness_values = []

    for swarm in population:
        for part in swarm:
            fitness = part.fitness.values[0]
            initial_fitness_values.append(fitness)

    print("\nInitial Fitness Values Before Optimization:")
    for i, fitness in enumerate(initial_fitness_values):
        print(f"Particle {i + 1}: Fitness = {fitness:.2f}")

    for iteration in range(max_iterations):
        rexcl = ((BOUNDS[1] - BOUNDS[0])) / (2 * len(population)**(1.0 / NDIM))
        print("Rexcl: ", rexcl)
        if verbose:
            print(f"Iteration {iteration + 1}/{max_iterations}")

        for i, swarm in enumerate(population):
            for part in swarm:
                if random.random() < 0.7:
                    apply_neighborhood_moves(part, data, constraints, days, periods) #neighborhood moves
                else:
                    updateParticle(data, part, part.best, swarm.best, chi, c1, c2, constraints) #swap

                # Re-evaluate the fitness after updating the particle
                part.fitness.values = toolbox.evaluate(part)

                if part.bestfit is None or part.fitness.values < part.bestfit.values:
                    part.best = toolbox.clone(part)
                    part.bestfit.values = part.fitness.values
                    print("Updated part bestfit:", part.bestfit.values)

                # Print current best in swarm and the current part fitness
                print("Swarm bestfit before comparison:", swarm.bestfit.values)
                print("Particle bestfit value: ", part.bestfit.values)
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
        print("Best fitness: ", best_fitness_in_population)

        # If global best fitness has changed, update global best and reevaluate personal bests
        if best_fitness_in_population < best_global_fitness:
            best_global_fitness = best_fitness_in_population
            # Update global best particle
            global_best_particle = toolbox.clone([swarm.best for swarm in population if swarm.bestfit.values[0] == best_fitness_in_population][0])

            # Track the iteration when the global best is updated
            last_global_best_update = iteration
            print(f"Global best updated at iteration {iteration+1}")

            # Re-evaluate personal bests for particles
            for swarm in population:
                for particle in swarm:
                    # Check if current particle's fitness is better than its personal best
                    if particle.fitness < particle.bestfit:
                        particle.best = toolbox.clone(particle)
                        particle.bestfit = particle.fitness

            print("Global best fitness updated.")

        # Stop if the fitness meets the target of 5 or less
        if best_global_fitness <= 0:
            print(f"\nStopping early as target fitness of 0 was reached: {best_global_fitness}")
            break

        # Exclusion mechanism
        room_map = {room["id"]: idx for idx, room in enumerate(rooms)}
        if len(population) > 1:
            reinit_swarms = set()
            for s1, s2 in itertools.combinations(range(len(population)), 2):
                if population[s1].best and population[s2].best:
                    distance = math.sqrt(
                        sum(
                            (entry1["day"] - entry2["day"])**2 +
                            (entry1["period"] - entry2["period"])**2 +
                            (room_map[entry1["room_id"]] - room_map[entry2["room_id"]])**2
                            for entry1, entry2 in zip(population[s1].best, population[s2].best)
                        )
                    )
                    print("Distance: ", distance)
                    if distance < rexcl:
                        print("Some swarm needs to be reinitialized.")
                        reinit_swarms.add(s1 if population[s1].bestfit <= population[s2].bestfit else s2)
            print(reinit_swarms)
            for s in reinit_swarms:
                print("Reinitializing Swarm: ", s)
                convertQuantum(
                    swarm, 
                    RCLOUD, 
                    swarm.best, 
                    min_distance, 
                    constraints, 
                    courses, 
                    curricula,
                    rooms
                )
                for part in population[s]:
                    part.fitness.values = toolbox.evaluate(part)

        not_converged = sum(
            1 for swarm in population if any(
                math.sqrt(
                    sum(
                        ((entry1["day"] - entry2["day"]) / days)**2 + 
                        ((entry1["period"] - entry2["period"]) / periods)**2 + 
                        ((room_map[entry1["room_id"]] - room_map[entry2["room_id"]]) / len(rooms))**2
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


    # Determine the best particle across all swarms based on bestfit values if available
    valid_swarms = [swarm for swarm in population if swarm.best is not None and swarm.bestfit is not None]

    print("\nOptimization Completed.")
    print("Initial Fitness Values Before Optimization:")
    for i, fitness in enumerate(initial_fitness_values):
        print(f"Particle {i + 1}: Fitness = {fitness:.2f}")

    if valid_swarms:
        # Find the swarm with the minimum bestfit value
        best_swarm = min(valid_swarms, key=lambda s: s.bestfit.values[0])
        final_best_schedule = [entry for entry in best_swarm.best]
        print("\nFinal Best Solution Found (Fitness):", best_swarm.bestfit.values[0])
    else:
        # Fallback if no swarm has a valid best (highly unlikely with this setup)
        final_best_schedule = None
        print("\nNo solution found.")

    print(f"The last global best was updated at iteration {last_global_best_update + 1}")
    return final_best_schedule