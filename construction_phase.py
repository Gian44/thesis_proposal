import random
from collections import defaultdict

def backtrack(schedule, used_slots, teacher_schedule, curriculum_schedule, unassigned_schedule, domains, cbs, courses, curricula, constraints, best_value):
    """
    Backtracking mechanism with Conflict-Based Statistics (CBS) to handle conflicts dynamically.
    Focuses only on courses conflicting with the problematic course's best candidate slot.
    """
    if not unassigned_schedule:
        raise Exception("[ERROR] Backtracking failed - no solution possible.")

    # Select the next course to process from the list of unassigned courses
    course_id = unassigned_schedule.pop(0)

    print(f"[DEBUG] Backtracking initiated for Course {course_id}")

    conflicts = []
    if best_value:
        print("The best value is: " + str(best_value))
        room, day, period = best_value

        # Find courses in conflict with this slot
        for entry in schedule:
            if has_conflict(entry, {"room_id": room, "day": day, "period": period, "course_id": course_id},
                            constraints, curriculum_schedule, teacher_schedule, used_slots, courses, curricula):
                conflicts.append(entry)
    else:
        # If no `best_value`, fallback to removing all courses that conflict with the problematic course
        for entry in schedule:
            if has_conflict(entry, course_id, constraints, curriculum_schedule, teacher_schedule, used_slots, courses, curricula):
                conflicts.append(entry)

    # Sort conflicts based on CBS (higher conflict frequency first)
    conflicts.sort(key=lambda conflict: cbs[(conflict["course_id"], (conflict["room_id"], conflict["day"], conflict["period"]))], reverse=True)

    # Remove only the conflicting courses from the schedule
    for conflict in conflicts:
        schedule.remove(conflict)
        used_slots.discard((conflict["room_id"], conflict["day"], conflict["period"]))
        teacher_schedule[get_teacher(courses, conflict["course_id"])].discard((conflict["day"], conflict["period"]))
        curriculum_ids = [
            curriculum["id"] for curriculum in curricula if conflict["course_id"] in curriculum["courses"]
        ]
        for curriculum_id in curriculum_ids:
            curriculum_schedule[curriculum_id].discard((conflict["day"], conflict["period"]))

        # Add the conflicting course back to the unassigned list
        unassigned_schedule.append(conflict["course_id"])

        # Update CBS for this conflict
        cbs[(conflict["course_id"], (conflict["room_id"], conflict["day"], conflict["period"]))] += 1

        print(f"[DEBUG] Removed conflicting Course {conflict['course_id']} from the schedule")

    # Add the problematic course back to unassigned_schedule to retry its assignment later
    unassigned_schedule.append(course_id)
    print(f"[DEBUG] Course {course_id} added back to the unassigned list for reassignment")

    # Debugging the count of unassigned slots
    print(f"[DEBUG] Number of unassigned slots after backtracking: {len(unassigned_schedule)}")

def ifs_generate(courses, rooms, num_days, periods_per_day, constraints, curricula):
    """
    Iterative Forward Search (IFS) with Conflict-Based Statistics (CBS) and Backtracking 
    to generate a feasible initial schedule that satisfies all hard constraints.
    """
    schedule = []
    used_slots = set()  # Track (room, day, period) to avoid conflicts
    curriculum_schedule = defaultdict(set)  # Tracks (day, period) for each curriculum
    teacher_schedule = defaultdict(set)  # Tracks (day, period) for each teacher
    cbs = defaultdict(int)  # Conflict-based statistics (CBS) to guide value selection

    # Precompute domains for each course
    domains = {
        course["id"]: [
            (room["id"], day, period)
            for room in rooms
            for day in range(num_days)
            for period in range(periods_per_day)
        ]
        for course in courses
    }

    # List of unassigned courses
    unassigned_schedule = [course["id"] for course in courses for _ in range(course["num_lectures"])]
    print("[DEBUG] Starting Iterative Forward Search (IFS)...")

    while unassigned_schedule:
        course_id = unassigned_schedule.pop(0)  # Select the next course to assign

        print(f"[DEBUG] Assigning Course {course_id}. Remaining unassigned: {len(unassigned_schedule)}")

        # Step 2: Value Selection
        best_value = None
        min_violations = float('inf')
        candidate_values = []

        # Filter only unused slots for this course
        for room, day, period in domains[course_id]:
            if (room, day, period) in used_slots:
                continue  # Skip slots that are already used

            # Check hard constraint violations if this value is assigned
            violations = calculate_hard_constraints(
                course_id,
                [(room, day, period)],
                constraints,
                curriculum_schedule,
                teacher_schedule,
                used_slots,
                courses,
                curricula
            )

            if violations < min_violations:
                min_violations = violations
                candidate_values = [(room, day, period)]
            elif violations == min_violations:
                candidate_values.append((room, day, period))

        best_value = random.choice(candidate_values)
        if not candidate_values or min_violations > 0:
            print(f"[DEBUG] No feasible value for Course {course_id} or violations > 0. Initiating backtracking...")
            unassigned_schedule.append(course_id)  # Add back to the unassigned list for retrying
            backtrack(schedule, used_slots, teacher_schedule, curriculum_schedule, unassigned_schedule,
                      domains, cbs, courses, curricula, constraints, best_value)
            continue  # Restart the while loop after backtracking

        # Randomly break ties among the best candidate values
        best_value = random.choice(candidate_values)
        room, day, period = best_value
        print(f"[DEBUG] Assigning Course {course_id} to Room {room}, Day {day}, Period {period}")

        # Assign the selected value
        schedule.append({
            "course_id": course_id,
            "room_id": room,
            "day": day,
            "period": period
        })
        used_slots.add((room, day, period))
        teacher_schedule[get_teacher(courses, course_id)].add((day, period))

        # Update curriculum schedule
        curriculum_ids = [
            curriculum["id"] for curriculum in curricula if course_id in curriculum["courses"]
        ]
        for curriculum_id in curriculum_ids:
            curriculum_schedule[curriculum_id].add((day, period))

    print("[DEBUG] Iterative Forward Search completed successfully.")
    return schedule

def calculate_hard_constraints(course_id, candidate_values, constraints, curriculum_schedule, teacher_schedule, used_slots, courses, curricula):
    """
    Calculate the number of hard constraint violations for a given course and candidate value.
    """
    violations = 0

    for room, day, period in candidate_values:
        #print(f"[DEBUG] Checking Course {course_id} at Room {room}, Day {day}, Period {period}")

        # Room occupancy: No two lectures in the same room at the same time
        if (room, day, period) in used_slots:
            violations += 1
            print(f"[DEBUG] Conflict: Room {room} already used at Day {day}, Period {period}")

        # Curriculum conflicts: No two lectures in the same curriculum can share the same time slot
        curriculum_ids = [
            curriculum["id"] for curriculum in curricula if course_id in curriculum["courses"]
        ]
        for curriculum_id in curriculum_ids:
            if (day, period) in curriculum_schedule[curriculum_id]:
                violations += 1
                print(f"[DEBUG] Conflict: Curriculum {curriculum_id} has a conflict at Day {day}, Period {period}")

        # Teacher availability: No teacher can teach multiple lectures at the same time
        teacher = get_teacher(courses, course_id)
        if (day, period) in teacher_schedule[teacher]:
            violations += 1
            print(f"[DEBUG] Conflict: Teacher {teacher} already scheduled at Day {day}, Period {period}")

        # Unavailability constraints: Course must not be assigned to unavailable slots
        for constraint in constraints:
            if constraint["course"] == course_id and constraint["day"] == day and constraint["period"] == period:
                violations += 1
                print(f"[DEBUG] Conflict: Unavailability for Course {course_id} at Day {day}, Period {period}")

    return violations


def has_conflict(entry, course_id, constraints, curriculum_schedule, teacher_schedule, used_slots, courses, curricula):
    """
    Determine if an assigned lecture conflicts with a specific course based on hard constraints.
    """
    room, day, period = entry["room_id"], entry["day"], entry["period"]

    #print(f"[DEBUG] Checking conflicts for Course {course_id} against Entry: Room {room}, Day {day}, Period {period}")

    # Check room occupancy conflict
    if (room, day, period) in used_slots:
        #print(f"[DEBUG] Conflict detected: Room {room} occupied at Day {day}, Period {period}")
        return True

    # Check curriculum conflict
    curriculum_ids = [
        curriculum["id"] for curriculum in curricula if entry["course_id"] in curriculum["courses"]
    ]
    for curriculum_id in curriculum_ids:
        if (day, period) in curriculum_schedule[curriculum_id]:
            #print(f"[DEBUG] Conflict detected: Curriculum {curriculum_id} conflict at Day {day}, Period {period}")
            return True

    # Check teacher conflict
    teacher = get_teacher(courses, entry["course_id"])
    if (day, period) in teacher_schedule[teacher]:
        #print(f"[DEBUG] Conflict detected: Teacher {teacher} conflict at Day {day}, Period {period}")
        return True

    # Check unavailability constraint
    for constraint in constraints:
        if constraint["course"] == entry["course_id"] and constraint["day"] == day and constraint["period"] == period:
            #print(f"[DEBUG] Conflict detected: Unavailability constraint for Course {entry['course_id']} at Day {day}, Period {period}")
            return True

    print(f"[DEBUG] No conflict detected for Course {course_id} against Entry: Room {room}, Day {day}, Period {period}")
    return False

def get_teacher(courses, course_id):
    """
    Retrieve the teacher for a given course ID.
    """
    for course in courses:
        if course["id"] == course_id:
            return course["teacher"]
    return None
