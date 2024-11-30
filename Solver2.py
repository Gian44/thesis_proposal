from ctt_parser import CTTParser
from Schedule import Schedule
from course_manager import CourseManager

def main():
    parser = CTTParser("input.ctt")
    data = parser.read_file()

    # Initialize the schedule and manager
    schedule = Schedule(data['courses'], data['rooms'], data['days'], data['periods_per_day'], data['constraints'], data['curricula'])
    manager = CourseManager(schedule)

    # Assign courses and evaluate feasibility
    manager.assign_lectures()
    if schedule.is_feasible():
        print("Feasible schedule generated.")
    else:
        print("Failed to generate a feasible schedule.")

if __name__ == "__main__":
    main()
