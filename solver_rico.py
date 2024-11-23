import random
import math
import copy
import os

class Course:
    def __init__(self, course_id, teacher, lectures, min_working_days, students):
        self.id = course_id
        self.teacher = teacher
        self.lectures = lectures
        self.min_working_days = min_working_days
        self.students = students

class Room:
    def __init__(self, room_id, capacity):
        self.id = room_id
        self.capacity = capacity

class Curriculum:
    def __init__(self, curriculum_id, courses):
        self.id = curriculum_id
        self.courses = courses

class Constraint:
    def __init__(self, course_id, day, period):
        self.course_id = course_id
        self.day = day
        self.period = period

class CourseAssignment:
    def __init__(self, course, room_id, day, period):
        self.course = course
        self.room_id = room_id
        self.day = day
        self.period = period

class TimetableSolution:
    def __init__(self):
        self.assignments = []

    def assign_course(self, course, room_id, day, period):
        self.assignments.append(CourseAssignment(course, room_id, day, period))

    def get_assignments_for_period(self, day, period):
        return [assignment for assignment in self.assignments if assignment.day == day and assignment.period == period]

    def is_course_scheduled_at(self, course_id, day, period):
        return any(
            assignment.course.id == course_id and assignment.day == day and assignment.period == period for assignment
            in self.assignments)

    def get_distinct_days_scheduled(self, course_id):
        return len(set(assignment.day for assignment in self.assignments if assignment.course.id == course_id))

    def get_rooms_used_by_course(self, course_id):
        return set(assignment.room_id for assignment in self.assignments if assignment.course.id == course_id)

    def get_random_assignment(self):
        return random.choice(self.assignments)

    def count_scheduled_lectures(self, course_id):
        """Count how many lectures are scheduled for a specific course."""
        return sum(1 for assignment in self.assignments if assignment.course.id == course_id)

    def remove_assignment(self, assignment):
        """Remove an assignment if it exists in the list."""
        if assignment in self.assignments:
            self.assignments.remove(assignment)
        else:
            print(f"Warning: Attempted to remove an assignment that was not found: {assignment}")

    def remove_course_assignments(self, course_id):
        self.assignments = [a for a in self.assignments if a.course.id != course_id]

class TimetableSolverSA:
    def __init__(self, days, periods_per_day, temperature=10.0, cooling_rate=0.95, temp_length_coef=7,
                 reheat_length_coef=7):
        self.days = days
        self.periods_per_day = periods_per_day
        self.temperature = temperature
        self.cooling_rate = cooling_rate
        self.temp_length_coef = temp_length_coef
        self.reheat_length_coef = reheat_length_coef
        self.courses = []
        self.rooms = []
        self.curricula = []
        self.constraints = []
        self.total_lecture_limit = 0

    def get_course_by_id(self, course_id):
        """Helper function to retrieve a course object by its ID."""
        for course in self.courses:
            if course.id == course_id:
                return course
        return None

    def parse_input_file(self, filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
            section = None
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("COURSES:"):
                    section = "COURSES"
                    continue
                elif line.startswith("ROOMS:"):
                    section = "ROOMS"
                    continue
                elif line.startswith("CURRICULA:"):
                    section = "CURRICULA"
                    continue
                elif line.startswith("UNAVAILABILITY_CONSTRAINTS:"):
                    section = "CONSTRAINTS"
                    continue
                elif line.startswith("END."):
                    break

                if section == "COURSES":
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    course = Course(parts[0], parts[1], int(parts[2]), int(parts[3]), int(parts[4]))
                    self.courses.append(course)
                    self.total_lecture_limit += course.lectures
                elif section == "ROOMS":
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    self.rooms.append(Room(parts[0], int(parts[1])))
                elif section == "CURRICULA":
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    curriculum_id = parts[0]
                    course_list = parts[2:]
                    self.curricula.append(Curriculum(curriculum_id, course_list))
                elif section == "CONSTRAINTS":
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    self.constraints.append(Constraint(parts[0], int(parts[1]), int(parts[2])))

    def is_feasible_assignment(self, course, day, period, room, solution):
        if any(constraint.course_id == course.id and constraint.day == day and constraint.period == period
               for constraint in self.constraints):
            return False
        for assignment in solution.get_assignments_for_period(day, period):
            if assignment.room_id == room.id or assignment.course.teacher == course.teacher:
                return False
        for curriculum in self.curricula:
            if course.id in curriculum.courses:
                if any(solution.is_course_scheduled_at(course_id, day, period) for course_id in curriculum.courses):
                    return False
        return True

    def calculate_penalty(self, solution):
        penalty = 0
        for assignment in solution.assignments:
            room = next((room for room in self.rooms if room.id == assignment.room_id), None)
            if room and assignment.course.students > room.capacity:
                penalty += (assignment.course.students - room.capacity)

        for course in self.courses:
            distinct_days = solution.get_distinct_days_scheduled(course.id)
            if distinct_days < course.min_working_days:
                penalty += 5 * (course.min_working_days - distinct_days)

        for curriculum in self.curricula:
            for day in range(self.days):
                assignments = [
                    a for p in range(self.periods_per_day)
                    for a in solution.get_assignments_for_period(day, p)
                    if a.course.id in curriculum.courses
                ]
                for i in range(len(assignments) - 1):
                    if assignments[i + 1].period - assignments[i].period > 1:
                        penalty += 2

        for course in self.courses:
            rooms_used = solution.get_rooms_used_by_course(course.id)
            penalty += len(rooms_used) - 1

        return penalty

    def generate_initial_solution(self):
        initial_solution = TimetableSolution()
        max_attempts_per_lecture = 300  # Limit attempts per lecture placement
        max_total_attempts = 1000  # Hard limit on total attempts per course
        global_lectures_scheduled = 0  # Track the total number of lectures scheduled
        lecture_counts = {course.id: 0 for course in self.courses}  # Track lectures for each course

        for course in self.courses:
            lectures_scheduled = 0
            total_attempts = 0  # Track total attempts for each course

            while lectures_scheduled < course.lectures and global_lectures_scheduled < self.total_lecture_limit and total_attempts < max_total_attempts:
                assigned = False
                attempts = 0

                while not assigned and attempts < max_attempts_per_lecture:
                    room = random.choice(self.rooms)
                    day = random.randint(0, self.days - 1)
                    period = random.randint(0, self.periods_per_day - 1)

                    # Check if this assignment is feasible
                    if self.is_feasible_assignment(course, day, period, room, initial_solution):
                        initial_solution.assign_course(course, room.id, day, period)
                        lectures_scheduled += 1
                        global_lectures_scheduled += 1
                        lecture_counts[course.id] += 1
                        assigned = True  # Lecture successfully assigned to a slot

                        if global_lectures_scheduled >= self.total_lecture_limit:
                            print("Global lecture limit reached. Stopping further assignments.")
                            return initial_solution
                    else:
                        # Implement swap mechanism
                        blocking_assignments = initial_solution.get_assignments_for_period(day, period)
                        for blocking_assignment in blocking_assignments:
                            if self.try_swap(course, blocking_assignment, day, period, room, initial_solution):
                                lectures_scheduled += 1
                                global_lectures_scheduled += 1
                                lecture_counts[course.id] += 1
                                assigned = True
                                break

                    attempts += 1
                    total_attempts += 1

                if not assigned:
                    print(
                        f"Warning: Could not find a feasible slot for lecture {lectures_scheduled + 1} of course {course.id} after {max_attempts_per_lecture} attempts.")

                    # Implement backtracking
                    if attempts >= max_attempts_per_lecture:
                        print(f"Backtracking for course {course.id} due to scheduling difficulty.")
                        initial_solution.remove_course_assignments(course.id)  # Remove all assignments for this course
                        lectures_scheduled = 0
                        global_lectures_scheduled -= lecture_counts[course.id]
                        lecture_counts[course.id] = 0
                        total_attempts += 1

            # Final check: If unable to schedule all lectures for the course
            if lectures_scheduled < course.lectures:
                print(
                    f"Error: Could not schedule all {course.lectures} lectures for course {course.id}. Only {lectures_scheduled} were scheduled.")

        print(f"Total lecture limit based on input instance: {self.total_lecture_limit}")
        print(f"Total lectures scheduled: {global_lectures_scheduled}")
        print(f"Lecture counts by course: {lecture_counts}")
        return initial_solution

    def try_swap(self, unassigned_course, blocking_assignment, day, period, room, solution):
        """Attempt to swap the blocking course with the unassigned course."""
        # Remove the blocking assignment temporarily
        solution.remove_assignment(blocking_assignment)

        # Check if the unassigned course can fit in the blocking assignment's slot without duplicating its timeslot
        if self.is_feasible_assignment(unassigned_course, day, period, room, solution) and not solution.is_course_scheduled_at(unassigned_course.id, day, period):
            # Assign the unassigned course to the blocking assignment's slot
            solution.assign_course(unassigned_course, room.id, day, period)

            # Attempt to reassign the blocking course to another slot
            reassigned = False
            for new_day in range(self.days):
                for new_period in range(self.periods_per_day):
                    for new_room in self.rooms:
                        # Check if the blocking course can be reassigned without duplicating its timeslot
                        if self.is_feasible_assignment(blocking_assignment.course, new_day, new_period, new_room, solution) and not solution.is_course_scheduled_at(blocking_assignment.course.id, new_day, new_period):
                            solution.assign_course(blocking_assignment.course, new_room.id, new_day, new_period)
                            reassigned = True
                            break
                    if reassigned:
                        break
                if reassigned:
                    break

            # If reassigning the blocking course fails, undo the swap
            if not reassigned:
                solution.remove_assignment(next(a for a in solution.assignments if a.course.id == unassigned_course.id))
                solution.assign_course(blocking_assignment.course, blocking_assignment.room_id, blocking_assignment.day, blocking_assignment.period)
                return False

            return True  # Swap was successful

        # Re-add the blocking assignment if the swap is not feasible
        solution.assign_course(blocking_assignment.course, blocking_assignment.room_id, blocking_assignment.day, blocking_assignment.period)
        return False



    def generate_neighbor(self, solution):
        neighbor = copy.deepcopy(solution)
        random_assignment = neighbor.get_random_assignment()

        new_day = random.randint(0, self.days - 1)
        new_period = random.randint(0, self.periods_per_day - 1)
        new_room = random.choice(self.rooms)

        if self.is_feasible_assignment(random_assignment.course, new_day, new_period, new_room, neighbor):
            neighbor.assign_course(random_assignment.course, new_room.id, new_day, new_period)

        return neighbor

    def solve(self, max_iterations=1000, min_temperature=1e-10):
        current_solution = self.generate_initial_solution()
        best_solution = copy.deepcopy(current_solution)
        current_penalty = self.calculate_penalty(current_solution)
        best_penalty = current_penalty

        iteration = 0
        while iteration < max_iterations and self.temperature > min_temperature:
            candidate_solution = self.generate_neighbor(current_solution)
            candidate_penalty = self.calculate_penalty(candidate_solution)

            if self.accept_solution(candidate_penalty, current_penalty):
                current_solution = candidate_solution
                current_penalty = candidate_penalty

                if candidate_penalty < best_penalty:
                    best_solution = copy.deepcopy(candidate_solution)
                    best_penalty = candidate_penalty

            self.temperature *= self.cooling_rate

            if iteration % self.reheat_length_coef == 0:
                self.temperature /= self.cooling_rate

            iteration += 1

        return best_solution

    def accept_solution(self, candidate_penalty, current_penalty):
        if candidate_penalty < current_penalty:
            return True
        return random.random() < math.exp((current_penalty - candidate_penalty) / self.temperature)

def save_solution_to_files(solution, directory, lecture_limit):
    out_filepath = os.path.join(directory, "solution.out")
    txt_filepath = os.path.join(directory, "solution.txt")
    with open(out_filepath, 'w') as out_file, open(txt_filepath, 'w') as txt_file:
        for assignment in solution.assignments[:lecture_limit]:
            line = f"{assignment.course.id}\t{assignment.room_id}\t{assignment.day}\t{assignment.period}\n"
            out_file.write(line)
            txt_file.write(line)
    print(f"Solution saved to {out_filepath} and {txt_filepath}")

# Main execution
solver = TimetableSolverSA(days=5, periods_per_day=6)
solver.parse_input_file("input1.txt")
best_solution = solver.solve()

output_directory = r"C:\Users\Gian\Desktop\Project\thesis_proposal"
save_solution_to_files(best_solution, output_directory, solver.total_lecture_limit)