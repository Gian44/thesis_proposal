import random
import os
import shutil
from parser import CTTParser


class Timetable:
    def __init__(self, courses, rooms, days, periods_per_day):
        self.courses = courses
        self.rooms = rooms
        self.days = days
        self.periods_per_day = periods_per_day
        self.timetable = self._initialize_timetable()

    def _initialize_timetable(self):
        return {
            day: {period: {room: -1 for room in self.rooms} for period in range(self.periods_per_day)}
            for day in range(self.days)
        }

    def reset(self):
        for day in self.timetable:
            for period in self.timetable[day]:
                for room in self.timetable[day][period]:
                    self.timetable[day][period][room] = -1
        for course in self.courses:
            self.courses[course]['assigned_lectures'] = 0

    def assign_course(self, course, slot):
        day, period, room = slot
        self.timetable[day][period][room] = course
        self.courses[course]['assigned_lectures'] += 1

    def get_assigned_courses_by_period(self, day, period):
        return [
            self.timetable[day][period][room]
            for room in self.timetable[day][period]
            if self.timetable[day][period][room] != -1
        ]

    def get_available_slots(self, course, unavailability_constraints):
        available_slots = []
        for day in self.timetable:
            for period in self.timetable[day]:
                if (day, period) in unavailability_constraints.get(course, []):
                    continue
                for room in self.timetable[day][period]:
                    if self.timetable[day][period][room] == -1:
                        available_slots.append((day, period, room))
        return available_slots


class ConstraintsManager:
    @staticmethod
    def has_conflict(course1, course2, courses, curricula):
        # Check if courses have the same teacher
        if courses[course1]['teacher'] == courses[course2]['teacher']:
            return True

        # Check if courses are in the same curriculum
        for curriculum_id, course_list in curricula.items():
            if course1 in course_list and course2 in course_list:
                return True

        return False


class FileManager:
    @staticmethod
    def delete_all_files_in_output(output_folder="output"):
        if os.path.exists(output_folder) and os.path.isdir(output_folder):
            for filename in os.listdir(output_folder):
                file_path = os.path.join(output_folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")

    @staticmethod
    def save_solution(timetable, filename="mnt\data\solution.out"):
        with open(filename, "w") as file:
            for day in timetable:
                for period in timetable[day]:
                    for room, course in timetable[day][period].items():
                        if course != -1:
                            file.write(f"{course} {room} {day} {period}\n")


def assign_courses(timetable, courses, unavailability_constraints, curricula):
    sequenced_courses = sorted(courses.keys(), key=lambda c: -len(unavailability_constraints.get(c, [])))

    for course in sequenced_courses:
        for _ in range(courses[course]['lectures'] - courses[course]['assigned_lectures']):
            available_slots = timetable.get_available_slots(course, unavailability_constraints)
            if available_slots:
                slot = random.choice(available_slots)
                timetable.assign_course(course, slot)

    return timetable.timetable


def main():
    # Parse the .ctt file
    parser = CTTParser("mnt/data/comp01.ctt")
    parser.parse()
    courses, rooms, unavailability_constraints, curricula, days, periods_per_day = parser.get_data()

    # Initialize timetable
    timetable = Timetable(courses, rooms, days, periods_per_day)

    # Assign courses
    final_timetable = assign_courses(timetable, courses, unavailability_constraints, curricula)

    # Save the solution
    FileManager.save_solution(final_timetable)


if __name__ == "__main__":
    main()
