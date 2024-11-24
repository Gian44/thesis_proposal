import random
from collections import defaultdict

def backtrack(schedule, used_slots, teacher_schedule, curriculum_schedule, unassigned_schedule, domains, courses, curricula, constraints, course_id, best_value, lecture_counts):
    """
    Backtracking mechanism to handle conflicts dynamically.
    """
    if not best_value:
        raise Exception("[ERROR] No best value provided for backtracking.")

    print(f"[DEBUG] Backtracking initiated for Course {course_id} with Best Value {best_value}")

    room, day, period = best_value
    conflicts = []

    # Find conflicting courses based on best_value
    proposed_assignment = {"room_id": room, "day": day, "period": period, "course_id": course_id}
    for entry in schedule:
        if has_conflict(entry, proposed_assignment, constraints, curriculum_schedule, teacher_schedule, used_slots, courses, curricula):
            conflicts.append(entry)

    # Remove conflicting courses
    for conflict in conflicts:
        schedule.remove(conflict)
        used_slots.discard((conflict["room_id"], conflict["day"], conflict["period"]))
        teacher_schedule[get_teacher(courses, conflict["course_id"])].discard((conflict["day"], conflict["period"]))
        curriculum_ids = [
            curriculum["id"] for curriculum in curricula if conflict["course_id"] in curriculum["courses"]
        ]
        for curriculum_id in curriculum_ids:
            curriculum_schedule[curriculum_id].discard((conflict["day"], conflict["period"]))

        # Re-add conflicting course to unassigned_schedule if its required lectures aren't fully assigned
        lecture_counts[conflict["course_id"]] -= 1  # Adjust lecture count for removed assignment
        if lecture_counts[conflict["course_id"]] < next(course["num_lectures"] for course in courses if course["id"] == conflict["course_id"]):
            unassigned_schedule.append(conflict["course_id"])
            print(f"[DEBUG] Removed conflicting Course {conflict['course_id']} from the schedule and re-added to unassigned.")

    # Re-add the problematic course only if it hasn't reached its required lectures
    if lecture_counts[course_id] < next(course["num_lectures"] for course in courses if course["id"] == course_id):
        unassigned_schedule.append(course_id)
        print(f"[DEBUG] Problematic Course {course_id} added back to the unassigned list.")

    # Debugging the count of unassigned slots
    print(f"[DEBUG] Number of unassigned slots after backtracking: {len(unassigned_schedule)}")

def ifs_generate(courses, rooms, num_days, periods_per_day, constraints, curricula):
    """
    Iterative Forward Search (IFS) to generate a feasible initial schedule.
    """
    schedule = []
    used_slots = set()  # Track (room, day, period) to avoid conflicts
    curriculum_schedule = defaultdict(set)  # Tracks (day, period) for each curriculum
    teacher_schedule = defaultdict(set)  # Tracks (day, period) for each teacher
    cbs = defaultdict(int)  # Conflict-based statistics
    lecture_counts = {course["id"]: 0 for course in courses}  # Track assigned lectures

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
        best_value, min_violations = select_value(
            course_id, domains, constraints, curriculum_schedule, teacher_schedule, used_slots, courses, curricula, cbs
        )
        print("MIN_VIOLATION:" + str(min_violations))
        # If no valid value found, initiate backtracking
        if not best_value or min_violations > 0:
            print(f"[DEBUG] No feasible value for Course {course_id}. Initiating backtracking...")
            backtrack(
                schedule, used_slots, teacher_schedule, curriculum_schedule, unassigned_schedule,
                domains, courses, curricula, constraints, course_id, best_value, lecture_counts
            )
            continue

        # Assign the selected value
        room, day, period = best_value
        print(f"[DEBUG] Assigning Course {course_id} to Room {room}, Day {day}, Period {period}")
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

        # Increment lecture count and ensure the course is not re-added to unassigned_schedule
        lecture_counts[course_id] += 1
        if lecture_counts[course_id] >= next(course["num_lectures"] for course in courses if course["id"] == course_id):
            print(f"[DEBUG] Course {course_id} fully assigned. Skipping reassignment.")

        print(f"[DEBUG] Remaining unassigned schedule: {unassigned_schedule}")

    print("[DEBUG] Iterative Forward Search completed successfully.")
    return schedule


def select_variable(unassigned, domains, constraints, curriculum_schedule, teacher_schedule, used_slots, curricula, courses):
    """
    Selects the next variable (course) to assign based on difficulty.
    """
    return min(
        unassigned,
        key=lambda course_id: (
            len(domains[course_id]),  # Smaller domains are prioritized
            calculate_hard_constraints(course_id, domains[course_id], constraints, curriculum_schedule, teacher_schedule, used_slots, courses, curricula)
        )
    )


def select_value(course_id, domains, constraints, curriculum_schedule, teacher_schedule, used_slots, courses, curricula, cbs):
    """
    Select the best value to assign to a course. If the best value results in violations,
    the calling function should trigger backtracking.
    """
    candidate_values = []
    min_violations = float('inf')

    # Evaluate all possible values in the domain
    for room, day, period in domains[course_id]:
        if (room, day, period) in used_slots:
            continue  # Skip already-used slots

        # Calculate violations for this value
        violations = calculate_hard_constraints(
            course_id, [(room, day, period)], constraints,
            curriculum_schedule, teacher_schedule, used_slots, courses, curricula
        )

        # Include Conflict-Based Statistics (CBS) to penalize repetitive conflicts
        violations += cbs[(course_id, (room, day, period))]

        # Track the best candidate values
        if violations < min_violations:
            min_violations = violations
            candidate_values = [(room, day, period)]
        elif violations == min_violations:
            candidate_values.append((room, day, period))

    if candidate_values:
        # Select one of the best candidates randomly
        best_value = random.choice(candidate_values)
        print(f"[DEBUG] Best value for Course {course_id} is {best_value} with {min_violations} violations.")
        return best_value, min_violations
    else:
        print(f"[ERROR] No valid values found for Course {course_id}.")
        return None, float('inf')


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
            #print(f"[DEBUG] Conflict: Room {room} already used at Day {day}, Period {period}")

        # Curriculum conflicts: No two lectures in the same curriculum can share the same time slot
        curriculum_ids = [
            curriculum["id"] for curriculum in curricula if course_id in curriculum["courses"]
        ]
        for curriculum_id in curriculum_ids:
            if (day, period) in curriculum_schedule[curriculum_id]:
                violations += 1
                #print(f"[DEBUG] Conflict: Curriculum {curriculum_id} has a conflict at Day {day}, Period {period}")

        # Teacher availability: No teacher can teach multiple lectures at the same time
        teacher = get_teacher(courses, course_id)
        if (day, period) in teacher_schedule[teacher]:
            violations += 1
            #print(f"[DEBUG] Conflict: Teacher {teacher} already scheduled at Day {day}, Period {period}")

        # Unavailability constraints: Course must not be assigned to unavailable slots
        for constraint in constraints:
            if constraint["course"] == course_id and constraint["day"] == day and constraint["period"] == period:
                violations += 1
                #print(f"[DEBUG] Conflict: Unavailability for Course {course_id} at Day {day}, Period {period}")

    return violations


def has_conflict(entry, proposed_assignment, constraints, curriculum_schedule, teacher_schedule, used_slots, courses, curricula):
    """
    Determine if an assigned lecture (entry) conflicts with the proposed assignment of the problematic course.

    Args:
        entry (dict): A scheduled course (already assigned).
        proposed_assignment (dict): Proposed assignment of the problematic course {room_id, day, period, course_id}.
    """
    # Extract details of the proposed assignment
    proposed_room = proposed_assignment["room_id"]
    proposed_day = proposed_assignment["day"]
    proposed_period = proposed_assignment["period"]
    proposed_course = proposed_assignment["course_id"]

    # Extract details of the current scheduled course
    scheduled_room = entry["room_id"]
    scheduled_day = entry["day"]
    scheduled_period = entry["period"]
    scheduled_course = entry["course_id"]

    # Room occupancy: Same room, day, and period
    if proposed_room == scheduled_room and proposed_day == scheduled_day and proposed_period == scheduled_period:
        return True

    # Curriculum conflict: Check if the courses share curricula and overlap in time
    scheduled_curricula = [
        curriculum["id"] for curriculum in curricula if scheduled_course in curriculum["courses"]
    ]
    proposed_curricula = [
        curriculum["id"] for curriculum in curricula if proposed_course in curriculum["courses"]
    ]
    # If any curricula overlap and timeslots match, it's a conflict
    if any(curriculum_id in scheduled_curricula for curriculum_id in proposed_curricula) and \
       scheduled_day == proposed_day and scheduled_period == proposed_period:
        return True

    # Teacher availability: Check if the same teacher is teaching both courses at the same time
    scheduled_teacher = get_teacher(courses, scheduled_course)
    proposed_teacher = get_teacher(courses, proposed_course)
    if scheduled_teacher == proposed_teacher and scheduled_day == proposed_day and scheduled_period == proposed_period:
        return True

    # Unavailability constraints: Check if the proposed course violates its unavailability
    for constraint in constraints:
        if constraint["course"] == proposed_course and constraint["day"] == proposed_day and constraint["period"] == proposed_period:
            return True

    return False


def get_teacher(courses, course_id):
    """
    Retrieve the teacher for a given course ID.
    """
    for course in courses:
        if course["id"] == course_id:
            return course["teacher"]
    return None
