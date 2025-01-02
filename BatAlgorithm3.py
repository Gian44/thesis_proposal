import copy
import random
import math

class BatAlgorithm:
    def __init__(self, courses, rooms, days, periods_per_day, curricula, unavailability_constraints, initial_population, max_iterations, alpha=0.9, gamma=0.9):
        self.courses = courses
        self.rooms = rooms
        self.days = days
        self.periods_per_day = periods_per_day
        self.curricula = curricula
        self.unavailability_constraints = unavailability_constraints
        self.initial_population = initial_population
        self.max_iterations = max_iterations
        self.population_size = len(initial_population)
        self.alpha = alpha  # Loudness reduction factor
        self.gamma = gamma  # Pulse emission increase factor

        self.best_solution = None
        self.best_fitness = float('inf')
        self.bats = self.initialize_population()

    def initialize_population(self):
        """Initialize the population of bats."""
        index = 0
        bats = []
        for solution in self.initial_population:
            bat_solution = copy.deepcopy(solution)
            bat = {
                'solution': bat_solution,
                'loudness': random.uniform(0, 1),
                'pulse_rate': random.uniform(0, 1),
                'velocity': random.randint(1, 10),
                'frequency': random.uniform(0, 1),
                'fitness': self.fitness(bat_solution)
            }
            bats.append(bat)
            
            # Display bats initial fitness
            print(f"Bat {index+1} Fitness {bat['fitness']}")
            index += 1

            # Update the global best solution
            if bat['fitness'] < self.best_fitness:
                self.best_fitness = bat['fitness']
                self.best_solution = (bat_solution)
        return bats

    def is_feasible(self, solution):
        """Check if the solution satisfies all hard constraints."""

        # 1. All lectures of a course must be scheduled in distinct periods (days and timeslots)
        course_periods = set()  # Tracks (course, day, timeslot) assignments
        for course, assignments in solution.items():
            for _, day, timeslot in assignments:
                if (course, day, timeslot) in course_periods:
                    return False  # A course has more than one lecture in the same period
                course_periods.add((course, day, timeslot))
                
        # Prevent multiple lectures of the same course being assigned to different rooms at the same day and timeslot
        for course, assignments in solution.items():
            period_assignments = set()
            for room, day, timeslot in assignments:
                if (day, timeslot) in period_assignments:
                    return False  # Multiple lectures of the same course assigned to different rooms at the same period
                period_assignments.add((day, timeslot))

        # 2. No two lectures in the same room at the same time
        room_periods = set()  # Tracks (room, day, timeslot) assignments
        for course, assignments in solution.items():
            for room, day, timeslot in assignments:
                if (room, day, timeslot) in room_periods:
                    return False  # A room has more than one lecture in the same period
                room_periods.add((room, day, timeslot))

        # 3. Conflict avoidance: No overlapping lectures for the same teacher or curricula
        teacher_periods = set()  # Tracks (teacher, day, timeslot) assignments
        curriculum_periods = set()  # Tracks (curriculum, day, timeslot) assignments
        for course, assignments in solution.items():
            teacher = self.courses[course]['teacher']
            for _, day, timeslot in assignments:
                # Check teacher conflicts
                if (teacher, day, timeslot) in teacher_periods:
                    return False  # The teacher is assigned to multiple lectures at the same period
                teacher_periods.add((teacher, day, timeslot))

                # Check curriculum conflicts
                for curriculum, courses_in_cur in self.curricula.items():
                    if course in courses_in_cur:
                        if (curriculum, day, timeslot) in curriculum_periods:
                            return False  # The curriculum has overlapping lectures
                        curriculum_periods.add((curriculum, day, timeslot))

        # 4. Teacher availability constraints
        for course, assignments in solution.items():
            for _, day, timeslot in assignments:
                if course in self.unavailability_constraints and (day, timeslot) in self.unavailability_constraints[course]:
                    return False  # The course is assigned to an unavailable period

        
        return True  # If no violations, the solution is feasible

    def fitness(self, solution):
        """Calculate fitness based on soft constraint violations."""
        costs = {
            "room_capacity": 0,
            "min_working_days": 0,
            "room_stability": 0,
            "curriculum_compactness": 0
        }

        # Check soft constraints
        for course, assignments in solution.items():
            # Room capacity cost
            for room, _, _ in assignments:
                room_capacity_cost = max(0, self.courses[course]['students'] - self.rooms[room])
                costs["room_capacity"] += room_capacity_cost
            
            # Minimum working days cost
            unique_days = set(day for _, day, _ in assignments)
            missing_days = max(0, self.courses[course]['min_days'] - len(unique_days))
            costs["min_working_days"] += missing_days * 5  # Penalty is 5 points per missing day

            # Room stability cost
            used_rooms = set(room for room, _, _ in assignments)
            room_stability_cost = len(used_rooms) - 1
            costs["room_stability"] += room_stability_cost

        # Curriculum compactness cost
        for curriculum, courses_in_cur in self.curricula.items():
            for day in range(self.days):  # Use self.days for correct day range
                periods_with_lectures = []
                for course in courses_in_cur:
                    if course in solution:
                        periods_with_lectures.extend(
                            timeslot for _, d, timeslot in solution[course] if d == day
                        )
                # Sort periods to identify gaps
                periods_with_lectures = sorted(periods_with_lectures)
                isolated_lectures = 0
                if periods_with_lectures:
                    for i in range(len(periods_with_lectures)):
                        # Check if the current lecture is isolated
                        is_isolated = (
                            (i == 0 or periods_with_lectures[i] != periods_with_lectures[i - 1] + 1) and
                            (i == len(periods_with_lectures) - 1 or periods_with_lectures[i] != periods_with_lectures[i + 1] - 1)
                        )
                        if is_isolated:
                            isolated_lectures += 1
                costs["curriculum_compactness"] += isolated_lectures * 2  # Penalty is 2 points per isolated lecture

        # Total cost
        total_cost = sum(costs.values())
        return total_cost

    def move(self, solution):
        """Generate a local solution around the current solution of the bat."""
        candidate_solution = copy.deepcopy(solution)
        
        # ROOM DAY TIMESLOT

        # Perform 1-0 moves: change room, day, timeslot combination
        course1 = random.choice(list(candidate_solution.keys()))

        idx1 = random.randint(0, len(candidate_solution[course1]) - 1)
        room1, day1, timeslot1 = candidate_solution[course1][idx1]

        # Randomly perturb room, day, timeslot
        new_room = random.choice(list(self.rooms.keys()))
        new_day = random.randint(0, self.days - 1)
        new_timeslot = random.randint(0, self.periods_per_day - 1)

        # Get the original assignment
        original_assignment = candidate_solution[course1][idx1]

        # Update the assignment
        candidate_solution[course1][idx1] = (new_room, new_day, new_timeslot)

        current_fitness = self.fitness(candidate_solution)
        # Validate feasibility
        if not self.is_feasible(candidate_solution) or current_fitness >= self.fitness(solution):
            candidate_solution[course1][idx1] = original_assignment  # Revert if infeasible
        else:
            print("perform 1-0 move room day timeslot")

        # Perform 1-1: Swap room, day and timeslot between two lectures
        course2, course3 = random.sample(list(candidate_solution.keys()), 2)

        idx2 = random.randint(0, len(candidate_solution[course2]) - 1)
        idx3 = random.randint(0, len(candidate_solution[course3]) - 1)

        # Get the assignments
        room2, day2, timeslot2 = candidate_solution[course2][idx2]
        room3, day3, timeslot3 = candidate_solution[course3][idx3]

        # Swap the room, day and timeslot
        candidate_solution[course2][idx2] = (room3, day3, timeslot3)
        candidate_solution[course3][idx3] = (room2, day2, timeslot2)

        current_fitness = self.fitness(candidate_solution)
        # Validate feasibility
        if not self.is_feasible(candidate_solution) or current_fitness >= self.fitness(solution):
            # Revert if infeasible
            candidate_solution[course2][idx2] = (room2, day2, timeslot2)
            candidate_solution[course3][idx3] = (room3, day3, timeslot3)
        else:
            print("perform 1-1 swap room, day timeslot")

        # DAY TIMESLOT

        # Perform 1-0 moves: change day, timeslot combination
        course1 = random.choice(list(candidate_solution.keys()))

        idx1 = random.randint(0, len(candidate_solution[course1]) - 1)
        room1, day1, timeslot1 = candidate_solution[course1][idx1]

        # Randomly perturb day, timeslot
        new_day = random.randint(0, self.days - 1)
        new_timeslot = random.randint(0, self.periods_per_day - 1)

        # Get the original assignment
        original_assignment = candidate_solution[course1][idx1]

        # Update the assignment
        candidate_solution[course1][idx1] = (room1, new_day, new_timeslot)

        current_fitness = self.fitness(candidate_solution)
        # Validate feasibility
        if not self.is_feasible(candidate_solution) or current_fitness >= self.fitness(solution):
            candidate_solution[course1][idx1] = original_assignment  # Revert if infeasible
        else:
            print("perform 1-0 move day timeslot")

        # Perform 1-1: Swap day and timeslot between two lectures
        course2, course3 = random.sample(list(candidate_solution.keys()), 2)

        idx2 = random.randint(0, len(candidate_solution[course2]) - 1)
        idx3 = random.randint(0, len(candidate_solution[course3]) - 1)

        # Get the assignments
        room2, day2, timeslot2 = candidate_solution[course2][idx2]
        room3, day3, timeslot3 = candidate_solution[course3][idx3]

        # Swap the day and timeslot
        candidate_solution[course2][idx2] = (room2, day3, timeslot3)
        candidate_solution[course3][idx3] = (room3, day2, timeslot2)

        current_fitness = self.fitness(candidate_solution)
        # Validate feasibility
        if not self.is_feasible(candidate_solution) or current_fitness >= self.fitness(solution):
            # Revert if infeasible
            candidate_solution[course2][idx2] = (room2, day2, timeslot2)
            candidate_solution[course3][idx3] = (room3, day3, timeslot3)
        else:
            print("perform 1-1 swap day timeslot")

        # DAY

        # Perform 1-0 moves: change day
        course1 = random.choice(list(candidate_solution.keys()))

        idx1 = random.randint(0, len(candidate_solution[course1]) - 1)
        room1, day1, timeslot1 = candidate_solution[course1][idx1]

        # Randomly perturb day
        new_day = random.randint(0, self.days - 1)

        # Get the original assignment
        original_assignment = candidate_solution[course1][idx1]

        # Update the assignment
        candidate_solution[course1][idx1] = (room1, new_day, timeslot1)

        current_fitness = self.fitness(candidate_solution)
        # Validate feasibility
        if not self.is_feasible(candidate_solution) or current_fitness >= self.fitness(solution):
            candidate_solution[course1][idx1] = original_assignment  # Revert if infeasible
        else:
            print("perform 1-0 move day")

        # Perform 1-1: Swap day between two lectures
        course1, course2 = random.sample(list(candidate_solution.keys()), 2)

        idx1 = random.randint(0, len(candidate_solution[course1]) - 1)
        idx2 = random.randint(0, len(candidate_solution[course2]) - 1)

        # Get the assignments
        room1, day1, timeslot1 = candidate_solution[course1][idx1]
        room2, day2, timeslot2 = candidate_solution[course2][idx2]

        # Swap day
        candidate_solution[course1][idx1] = (room1, day2, timeslot1)
        candidate_solution[course2][idx2] = (room2, day1, timeslot2)

        current_fitness = self.fitness(candidate_solution)
        # Validate feasibility
        if not self.is_feasible(candidate_solution) or current_fitness >= self.fitness(solution):
            # Revert if infeasible
            candidate_solution[course1][idx1] = (room1, day1, timeslot1)
            candidate_solution[course2][idx2] = (room2, day2, timeslot2)
        else:
            print("perform 1-1 swap day")

        # TIMESLOT

        # Perform 1-0 moves: change timeslot
        course1 = random.choice(list(candidate_solution.keys()))

        idx1 = random.randint(0, len(candidate_solution[course1]) - 1)
        room1, day1, timeslot1 = candidate_solution[course1][idx1]

        # Randomly perturb day, timeslot
        new_timeslot = random.randint(0, self.periods_per_day - 1)

        # Get the original assignment
        original_assignment = candidate_solution[course1][idx1]

        # Update the assignment
        candidate_solution[course1][idx1] = (room1, day1, new_timeslot)

        current_fitness = self.fitness(candidate_solution)
        # Validate feasibility
        if not self.is_feasible(candidate_solution) or current_fitness >= self.fitness(solution):
            candidate_solution[course1][idx1] = original_assignment  # Revert if infeasible
        else:
            print("perform 1-0 move timeslot")

        # Perform 1-1: Swap timeslot between two lectures
        course1, course2 = random.sample(list(candidate_solution.keys()), 2)

        idx1 = random.randint(0, len(candidate_solution[course1]) - 1)
        idx2 = random.randint(0, len(candidate_solution[course2]) - 1)

        # Get the assignments
        room1, day1, timeslot1 = candidate_solution[course1][idx1]
        room2, day2, timeslot2 = candidate_solution[course2][idx2]

        # Swap timeslot
        candidate_solution[course1][idx1] = (room1, day1, timeslot2)
        candidate_solution[course2][idx2] = (room2, day2, timeslot1)

        current_fitness = self.fitness(candidate_solution)
        # Validate feasibility
        if not self.is_feasible(candidate_solution) or current_fitness >= self.fitness(solution):
            # Revert if infeasible
            candidate_solution[course1][idx1] = (room1, day1, timeslot1)
            candidate_solution[course2][idx2] = (room2, day2, timeslot2)
        else:
            print("perform 1-1 swap timeslot")

        # ROOM and DAY

        # Perform 1-0 moves: change room and day
        course1 = random.choice(list(candidate_solution.keys()))

        idx1 = random.randint(0, len(candidate_solution[course1]) - 1)
        room1, day1, timeslot1 = candidate_solution[course1][idx1]

        # Randomly perturb room, day
        new_room = random.choice(list(self.rooms.keys()))
        new_day = random.randint(0, self.days - 1)

        # Get the original assignment
        original_assignment = candidate_solution[course1][idx1]

        # Update the assignment
        candidate_solution[course1][idx1] = (new_room, new_day, timeslot1)

        current_fitness = self.fitness(candidate_solution)
        # Validate feasibility
        if not self.is_feasible(candidate_solution) or current_fitness >= self.fitness(solution):
            candidate_solution[course1][idx1] = original_assignment  # Revert if infeasible
        else:
            print("perform 1-0 move room and day")

        # Perform 1-1: Swap room and day between two lectures
        course1, course2 = random.sample(list(candidate_solution.keys()), 2)

        idx1 = random.randint(0, len(candidate_solution[course1]) - 1)
        idx2 = random.randint(0, len(candidate_solution[course2]) - 1)

        # Get the assignments
        room1, day1, timeslot1 = candidate_solution[course1][idx1]
        room2, day2, timeslot2 = candidate_solution[course2][idx2]

        # Swap room and day
        candidate_solution[course1][idx1] = (room2, day2, timeslot1)
        candidate_solution[course2][idx2] = (room1, day1, timeslot2)

        current_fitness = self.fitness(candidate_solution)
        # Validate feasibility
        if not self.is_feasible(candidate_solution) or current_fitness >= self.fitness(solution):
            # Revert if infeasible
            candidate_solution[course1][idx1] = (room1, day1, timeslot1)
            candidate_solution[course2][idx2] = (room2, day2, timeslot2)
        else:
            print("perform 1-1 swap room and day")

        # ROOM and TIMESLOT

        # Perform 1-0 moves: change room and timeslot
        course1 = random.choice(list(candidate_solution.keys()))

        idx1 = random.randint(0, len(candidate_solution[course1]) - 1)
        room1, day1, timeslot1 = candidate_solution[course1][idx1]

        # Randomly perturb room, timeslot
        new_room = random.choice(list(self.rooms.keys()))
        new_timeslot = random.randint(0, self.periods_per_day - 1)

        # Get the original assignment
        original_assignment = candidate_solution[course1][idx1]

        # Update the assignment
        candidate_solution[course1][idx1] = (new_room, day1, new_timeslot)

        current_fitness = self.fitness(candidate_solution)
        # Validate feasibility
        if not self.is_feasible(candidate_solution) or current_fitness >= self.fitness(solution):
            candidate_solution[course1][idx1] = original_assignment  # Revert if infeasible
        else:
            print("perform 1-0 move room and timeslot")

        # Perform 1-1: Swap room and timeslot between two lectures
        course1, course2 = random.sample(list(candidate_solution.keys()), 2)

        idx1 = random.randint(0, len(candidate_solution[course1]) - 1)
        idx2 = random.randint(0, len(candidate_solution[course2]) - 1)

        # Get the assignments
        room1, day1, timeslot1 = candidate_solution[course1][idx1]
        room2, day2, timeslot2 = candidate_solution[course2][idx2]

        # Swap room and timeslot
        candidate_solution[course1][idx1] = (room2, day1, timeslot2)
        candidate_solution[course2][idx2] = (room1, day2, timeslot1)

        current_fitness = self.fitness(candidate_solution)
        # Validate feasibility
        if not self.is_feasible(candidate_solution) or current_fitness >= self.fitness(solution):
            # Revert if infeasible
            candidate_solution[course1][idx1] = (room1, day1, timeslot1)
            candidate_solution[course2][idx2] = (room2, day2, timeslot2)
        else:
            print("perform 1-1 swap room and timeslot")

        # ROOMS

        # Perform 1-0 moves: rooms
        course4 = random.choice(list(candidate_solution.keys()))

        idx4 = random.randint(0, len(candidate_solution[course4]) - 1)
        room4, day4, timeslot4 = candidate_solution[course4][idx4]

        # Randomly perturb room
        new_room = random.choice(list(self.rooms.keys()))

        # Get the original assignment
        original_assignment = candidate_solution[course4][idx4]

        # Update the assignment
        candidate_solution[course4][idx4] = (new_room, day4, timeslot4)

        current_fitness = self.fitness(candidate_solution)
        # Validate feasibility
        if not self.is_feasible(candidate_solution) or current_fitness >= self.fitness(solution):
            candidate_solution[course4][idx4] = original_assignment  # Revert if infeasible
        else:
            print("perform 1-0 move room")    
        
        # Perform 1-1: Swap rooms between two lectures
        course5, course6 = random.sample(list(candidate_solution.keys()), 2)

        idx5 = random.randint(0, len(candidate_solution[course5]) - 1)
        idx6 = random.randint(0, len(candidate_solution[course6]) - 1)

        # Get the assignments
        room5, day5, timeslot5 = candidate_solution[course5][idx5]
        room6, day6, timeslot6 = candidate_solution[course6][idx6]

        # Swap the rooms
        candidate_solution[course5][idx5] = (room6, day5, timeslot5)
        candidate_solution[course6][idx6] = (room5, day6, timeslot6)

        current_fitness = self.fitness(candidate_solution)
        # Validate feasibility
        if not self.is_feasible(candidate_solution) or current_fitness >= self.fitness(solution):
            # Revert if infeasible
            candidate_solution[course5][idx5] = (room5, day5, timeslot5)
            candidate_solution[course6][idx6] = (room6, day6, timeslot6)
        else:
            print("perform 1-1 swap room")
        
        return candidate_solution
    
    def numberOfMoves(self, bat, probability):
        rand = random.random()
        if rand <= probability:
            return bat['velocity']
        else:
            return abs(bat['fitness'] - self.best_fitness)
        

    def optimize(self):
        """Main optimization loop with an explicit exploitation stage."""

        for iteration in range(self.max_iterations):
            # Print iteration progress
            print(f"Iteration {iteration + 1}/{self.max_iterations}: Best Fitness = {self.best_fitness}")

            i = 1
            for bat in self.bats:
                # Update velocity based on fitness difference
                probability = bat['velocity'] / (bat['velocity'] + abs(bat['fitness'] - self.best_fitness))

                velocity = self.numberOfMoves(bat, probability)
                bat['velocity'] = velocity

                # Perform local search V (velocity) times
                for _ in range(velocity):
                    print(f"Bat {i} velocity (number of moves): {velocity}")
                    bat['solution'] = self.move(bat['solution'])
                    bat['fitness'] = self.fitness(bat['solution'])
                    print(f"Bat {i} fitness: {bat['fitness']}")

                rand = random.random()
                if rand > bat['pulse_rate']:
                    self.best_solution = self.move(self.best_solution)
                    self.best_fitness = self.fitness(self.best_solution)

                # Accept the new solution if fitness improves and rand < loudness
                if rand < bat['loudness'] and bat['fitness'] < self.best_fitness:
                    self.best_solution = bat['solution']
                    self.best_fitness = bat['fitness']

                    # Increase pulse rate and reduce loudness
                    bat['pulse_rate'] *= (1 - math.exp(-self.gamma * iteration))
                    bat['loudness'] *= self.alpha

                i += 1

        # Rank the bats and update the global best solution
        self.bats.sort(key=lambda x: x['fitness'])
        self.best_fitness = self.bats[0]['fitness']
        self.best_solution = self.bats[0]['solution']

        # Return the final best solution
        print("\nOptimization Complete!")
        print(f"Final Best Fitness: {self.best_fitness}")
        return self.best_solution