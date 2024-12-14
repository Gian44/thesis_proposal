import pandas as pd
import re
from multiswarm import main as multiswarm_optimization
from initialize_population2 import assign_courses  


# Step 1: Parse the .ctt file
def parse_ctt(file_path):
    with open(file_path, 'r') as file:
        data = {"courses": [], "rooms": [], "curricula": [], "constraints": []}
        section = None
        
        for line in file:
            line = line.strip()
            
            # General Information
            if line.startswith("Name:"):
                data["name"] = line.split(":")[1].strip()
            elif line.startswith("Courses:"):
                data["num_courses"] = int(line.split(":")[1].strip())
            elif line.startswith("Rooms:"):
                data["num_rooms"] = int(line.split(":")[1].strip())
            elif line.startswith("Days:"):
                data["num_days"] = int(line.split(":")[1].strip())
            elif line.startswith("Periods_per_day:"):
                data["periods_per_day"] = int(line.split(":")[1].strip())
            elif line.startswith("Curricula:"):
                data["num_curricula"] = int(line.split(":")[1].strip())
            elif line.startswith("Constraints:"):
                data["num_constraints"] = int(line.split(":")[1].strip())

            # Section Header Identifiers
            elif line == "COURSES:":
                section = "courses"
            elif line == "ROOMS:":
                section = "rooms"
            elif line == "CURRICULA:":
                section = "curricula"
            elif line == "UNAVAILABILITY_CONSTRAINTS:":
                section = "constraints"
            elif line == "END.":
                break

            # Section Parsing
            elif section == "courses":
                match = re.match(r"(\w+) (\w+) (\d+) (\d+) (\d+)", line)
                if match:
                    course = {"id": match.group(1), "teacher": match.group(2),
                              "num_lectures": int(match.group(3)), "min_days": int(match.group(4)),
                              "num_students": int(match.group(5))}
                    data["courses"].append(course)
                    
            elif section == "rooms":
                match = re.match(r"(\w+) (\d+)", line)
                if match:
                    room = {"id": match.group(1), "capacity": int(match.group(2))}
                    data["rooms"].append(room)
                    
            elif section == "curricula":
                parts = line.split()
                if len(parts) >= 3:  # Ensure there are at least an id, num_courses, and one course
                    curriculum = {"id": parts[0], "num_courses": int(parts[1]), "courses": parts[2:]}
                    data["curricula"].append(curriculum)
                else:
                    print(f"Warning: Skipping malformed curricula line: {line}")
                
            elif section == "constraints":
                match = re.match(r"(\w+) (\d+) (\d+)", line)
                if match:
                    constraint = {"course": match.group(1), "day": int(match.group(2)), "period": int(match.group(3))}
                    data["constraints"].append(constraint)
        return data

# Step 2: Generate Initial Feasible Solutions (Construction Phase)
def generate_initial_solution():
    initial_solution = assign_courses()
    return initial_solution

# Step 3: Optimize using Multi-Swarm PSO
def optimize_schedule(data):
    optimized_schedule = multiswarm_optimization(data, verbose=True)
    return optimized_schedule

# Step 4: Save Output to .csv and .out files
def save_output(schedule, csv_path, out_path):
    # Convert the schedule to DataFrame for CSV
    df = pd.DataFrame(schedule)
    df.to_csv(csv_path, index=False)
    
    # Write the output to .out format
    with open(out_path, 'w') as f:
        for entry in schedule:
            line = f"{entry['course_id']} {entry['room_id']} {entry['day']} {entry['period']}\n"
            f.write(line)

# Main execution
if __name__ == "__main__":
    # Parse the .ctt input file
    ctt_data = parse_ctt("mnt/data/comp02.ctt")

    # Generate initial feasible solution
    timetable = generate_initial_solution()
    #print("Initial Solution Generated")
    #print(initial_solution)
    with open("output/initial_solution.out", "w") as file:  # Open the file in write mode
            for day in timetable:
                for period in timetable[day]:
                    for room in timetable[day][period]:
                        if timetable[day][period][room] != -1:
                            file.write(f"{timetable[day][period][room]} {room} {day} {period}\n")

    # Save the initial solution
    #save_output(initial_solution, "mnt/data/solution.csv", "mnt/data/solution.out")

    # Optimize the solution using Multi-Swarm PSO
    optimized_solution = optimize_schedule(ctt_data)
    #print("Optimized Solution:")
    #print(optimized_solution)

    # Save the optimized solution
    save_output(optimized_solution, "mnt/data/comp02.csv", "mnt/data/comp02.out")
