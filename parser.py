class CTTParser:
    def __init__(self, filename):
        """
        Initializes the CTTParser with the given filename.
        Args:
            filename (str): Path to the .ctt file to be parsed.
        """
        self.filename = filename
        self.courses = {}
        self.rooms = {}
        self.unavailability_constraints = {}
        self.curricula = {}
        self.days = 0
        self.periods_per_day = 0

    def parse(self):
        """
        Parses the .ctt file and populates the class attributes.
        """
        with open(self.filename, 'r') as file:
            data = file.readlines()

        reading_section = None

        for line in data:
            line = line.strip()
            if not line or line == "END.":
                continue

            if line.startswith("Days"):
                parts = line.split()
                self.days = int(parts[1])

            if line.startswith("Periods_per_day:"):
                parts = line.split()
                self.periods_per_day = int(parts[1])

            if line.startswith("Name:") or line.startswith("Courses:") or line.startswith("Rooms:") or \
               line.startswith("Days:") or line.startswith("Periods_per_day:") or line.startswith("Constraints:"):
                continue

            if line == "COURSES:":
                reading_section = "COURSES"
                continue
            elif line == "ROOMS:":
                reading_section = "ROOMS"
                continue
            elif line == "UNAVAILABILITY_CONSTRAINTS:":
                reading_section = "CONSTRAINTS"
                continue
            elif line == "CURRICULA:":
                reading_section = "CURRICULA"
                continue

            if reading_section == "COURSES":
                self._parse_course(line)
            elif reading_section == "ROOMS":
                self._parse_room(line)
            elif reading_section == "CONSTRAINTS":
                self._parse_constraint(line)
            elif reading_section == "CURRICULA":
                self._parse_curriculum(line)

        self._initialize_course_metadata()

    def _parse_course(self, line):
        """
        Parses a course entry and adds it to the courses dictionary.
        Args:
            line (str): A line from the .ctt file in the COURSES section.
        """
        parts = line.split()
        course_id = parts[0]
        self.courses[course_id] = {
            'teacher': parts[1],
            'lectures': int(parts[2]),
            'min_days': int(parts[3]),
            'students': int(parts[4])
        }

    def _parse_room(self, line):
        """
        Parses a room entry and adds it to the rooms dictionary.
        Args:
            line (str): A line from the .ctt file in the ROOMS section.
        """
        parts = line.split()
        room_id = parts[0]
        self.rooms[room_id] = int(parts[1])

    def _parse_constraint(self, line):
        """
        Parses a constraint entry and adds it to the unavailability_constraints dictionary.
        Args:
            line (str): A line from the .ctt file in the UNAVAILABILITY_CONSTRAINTS section.
        """
        parts = line.split()
        course_id = parts[0]
        day = int(parts[1])
        period = int(parts[2])
        if course_id not in self.unavailability_constraints:
            self.unavailability_constraints[course_id] = []
        self.unavailability_constraints[course_id].append((day, period))

    def _parse_curriculum(self, line):
        """
        Parses a curriculum entry and adds it to the curricula dictionary.
        Args:
            line (str): A line from the .ctt file in the CURRICULA section.
        """
        parts = line.split()
        curriculum_id = parts[0]
        course_list = parts[2:]  # The list of courses in the curriculum
        self.curricula[curriculum_id] = course_list

    def _initialize_course_metadata(self):
        """
        Initializes metadata for each course, such as assigned lectures and assigned days.
        """
        for course_id in self.courses:
            self.courses[course_id]['assigned_lectures'] = 0

    def get_data(self):
        """
        Returns the parsed data as a tuple.
        Returns:
            tuple: Parsed courses, rooms, unavailability constraints, curricula, days, and periods per day.
        """
        return (self.courses, self.rooms, self.unavailability_constraints, self.curricula, self.days, self.periods_per_day)
