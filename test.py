from initialize_population import assign_courses
#from your_optimization_function import optimize_schedule

def save_timetable_to_file(timetable, filename):
    """
    Save the timetable to a file in the specified `.out` format.
    
    Args:
        timetable (dict): Final timetable in the format {day: {period: {room: course_id}}}.
        filename (str): Path to the output file.
    """
    with open(filename, 'w') as file:
        for day, periods in timetable.items():
            for period, rooms in periods.items():
                for room, course_id in rooms.items():
                    if course_id != -1:  # Only save non-empty slots
                        file.write(f"{course_id} {room} {day} {period}\n")

if __name__ == "__main__":
    ctt_file = "mnt\data\comp01.ctt"
    output_file = "mnt\data\solution.out"

    # Generate initial feasible population
    initial_timetable = assign_courses()

    # Optimize the timetable
    #optimized_timetable = optimize_schedule(initial_timetable)

    # Save the final optimized timetable to a file
    save_timetable_to_file(initial_timetable, output_file)
    print(f"Timetable saved to {output_file}")

