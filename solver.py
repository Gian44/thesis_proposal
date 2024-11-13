import pandas as pd
from multiswarm2 import main as multiswarm_optimization  
import re

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

# Step 2: Run Multi-Swarm Particle Swarm Optimization (MS-PSO)
def optimize_schedule(data):
    # Pass parsed data to multiswarm_optimization for use in generating and evaluating schedules
    schedule = multiswarm_optimization(data, verbose=True) 
    return schedule


# Step 3: Save Output to .csv and .out files
def save_output(schedule, csv_path, out_path):
    # Convert the schedule to DataFrame for CSV
    df = pd.DataFrame(schedule)  # Assume schedule is in tabular form with required columns
    df.to_csv(csv_path, index=False)
    
    # Write the output to .out format
    """Saves the schedule in the specified format to a .out file."""
    with open(out_path, 'w') as f:
        for entry in schedule:
            line = f"{entry['course_id']} {entry['room_id']} {entry['day']} {entry['period']}\n"
            f.write(line)

# Main execution
if __name__ == "__main__":
    # Parse the .ctt input file
    ctt_data = parse_ctt("mnt/data/comp01.ctt")

    # Run the optimization algorithm
    optimized_schedule = optimize_schedule(ctt_data)

    # Save the results
    save_output(optimized_schedule, "mnt/data/comp01_output.csv", "mnt/data/comp01_output.out")
