from constraints import HardConstraints, SoftConstraints

def evaluate(schedule, courses, rooms, curricula, constraints):
    hard_penalty = 0
    soft_penalty = 0

    for entry in schedule:
        # Hard Constraints
        if HardConstraints.check_room_occupancy(schedule, entry):
            hard_penalty += 5000
        if HardConstraints.check_teacher_availability(courses, schedule, entry):
            hard_penalty += 5000
        if HardConstraints.check_curriculum_conflicts(curricula, schedule, entry):
            hard_penalty += 5000

        # Soft Constraints
        soft_penalty += SoftConstraints.room_capacity_penalty(courses, rooms, entry)

    return hard_penalty + soft_penalty
