class HardConstraints:
    @staticmethod
    def check_room_occupancy(timetable, entry):
        room = entry["room_id"]
        day = entry["day"]
        period = entry["period"]
        if timetable[day][period][room] != -1:
            return True
        return False

    @staticmethod
    def check_teacher_availability(courses, schedule, entry):
        teacher = courses[entry["course_id"]]["teacher"]
        for other in schedule:
            if other["course_id"] == entry["course_id"]:
                continue
            if other["day"] == entry["day"] and other["period"] == entry["period"]:
                if teacher == courses[other["course_id"]]["teacher"]:
                    return True
        return False

    @staticmethod
    def check_curriculum_conflicts(curricula, schedule, entry):
        for curriculum in curricula:
            if entry["course_id"] in curriculum["courses"]:
                for other in schedule:
                    if other["course_id"] == entry["course_id"]:
                        continue
                    if other["day"] == entry["day"] and other["period"] == entry["period"]:
                        if other["course_id"] in curriculum["courses"]:
                            return True
        return False


class SoftConstraints:
    @staticmethod
    def room_capacity_penalty(courses, rooms, entry):
        room_capacity = rooms[entry["room_id"]]["capacity"]
        course_students = courses[entry["course_id"]]["num_students"]
        return max(0, course_students - room_capacity)

    @staticmethod
    def curriculum_compactness_penalty(curricula, schedule):
        penalty = 0
        for curriculum in curricula:
            courses = curriculum["courses"]
            for day, periods in schedule.items():
                for period, room in periods.items():
                    if any(course in courses for course in room.values()):
                        penalty += 1
        return penalty
