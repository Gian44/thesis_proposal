import random
from ctt_parser import read_ctt_file
import os
import shutil

# Read data from .ctt file
filename = 'mnt/data/comp01.ctt'  # Replace with your .ctt file name
courses, rooms, unavailability_constraints, curricula, days, periods_per_day= read_ctt_file(filename)
# Initialize timetable: days x periods_per_day x rooms
# -1 indicates empty slots
timetable = {
        day: {period: {room: -1 for room in rooms} for period in range(periods_per_day)}
        for day in range(days)
    }

def reset_timetable():
    for day in timetable:
        for period in timetable[day]:
            for room in timetable[day][period]:
                timetable[day][period][room] = -1
    for course in courses:
        courses[course]['assigned_lectures'] = 0

def delete_all_files_in_output():
    output_folder = "output"
    # Check if the folder exists
    if os.path.exists(output_folder) and os.path.isdir(output_folder):
        # Loop through and delete all files
        for filename in os.listdir(output_folder):
            file_path = os.path.join(output_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file or symlink
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directory
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

def get_available_slots(course, constraint_period=[-1,-1,-1]):
    available_slots = []
    for day in timetable:
        for period in timetable[day]:
            hasConflict = False
            if day == constraint_period[0] and period == constraint_period[1]: hasConflict = True
            for target_course in get_assigned_courses_by_period(day, period):
                if has_conflict(course, target_course):
                    hasConflict = True
                    break
            if hasConflict != True:
                for room in timetable[day][period]:
                    slot = [day, period, room]
                    isValid = True
                    if course in unavailability_constraints and (day, period) in unavailability_constraints[course]:
                        isValid = False
                    if timetable[day][period][room] == -1 and isValid:
                        available_slots.append(slot)
    return available_slots

def get_available_slots_with_conflict_course(course):
    available_slots = []
    conflict_courses = []
    for day in timetable:
        for period in timetable[day]:
            hasConflict = 0
            for target_course in get_assigned_courses_by_period(day, period):
                if has_conflict(course, target_course):
                    hasConflict += 1
            if hasConflict == 1:
                for room in timetable[day][period]:
                    slot = [day, period, room]
                    isValid = True
                    if course in unavailability_constraints and (day, period) in unavailability_constraints[course]:
                        isValid = False
                    if timetable[day][period][room] == -1 and isValid:
                        available_slots.append(slot)
                        conflict_courses.append(find_conflict_course(course, day, period))
    return available_slots, conflict_courses

def get_swappable_slots(course, constraint_period=[-1,-1,-1]):
    available_slots = []
    for day in timetable:
        for period in timetable[day]:
            hasConflict = 0
            conflict_course = ""
            if day == constraint_period[0] and period == constraint_period[1]: hasConflict = True
            for target_course in get_assigned_courses_by_period(day, period):
                    if has_conflict(course, target_course):
                        conflict_course = target_course
                        hasConflict += 1
            if hasConflict <= 1:
                for room in timetable[day][period]:
                    if has_conflict == 0 or timetable[day][period][room] == conflict_course: 
                        slot = [day, period, room]
                        isValid = True
                        if course in unavailability_constraints and (day, period) in unavailability_constraints[course]:
                            isValid = False
                            break
                        if timetable[day][period][room] != -1 and isValid:
                            if get_available_slots(timetable[day][period][room], [day, period, room]):
                                available_slots.append(slot)
    return available_slots

def get_overridable_slots(course):
    available_slots = []
    conflict_courses = []
    for day in timetable:
        for period in timetable[day]:
            hasConflict = 0
            for target_course in get_assigned_courses_by_period(day, period):
                if has_conflict(course, target_course):
                    hasConflict += 1
            if hasConflict == 1:
                for room in timetable[day][period]:
                    slot = [day, period, room]
                    isValid = True
                    if course in unavailability_constraints and (day, period) in unavailability_constraints[course]:
                        isValid = False
                    if isValid:
                        conflict_course = find_conflict_course(course, day, period)
                        if get_available_slots(conflict_course, [day, period, room]) and timetable[day][period][room] != -1:
                            if get_available_slots(timetable[day][period][room], [day, period, room]) and get_room(day, period, conflict_course) != room:
                                available_slots.append(slot)
                                conflict_courses.append(conflict_course)
    return available_slots, conflict_courses


def find_conflict_course(course, day, period):
    for target_course in get_assigned_courses_by_period(day, period):
        if has_conflict(course, target_course):
            return target_course
    return -1

def has_conflict(course1, course2):
    # Check if courses have the same teacher
    if courses[course1]['teacher'] == courses[course2]['teacher']:
        return True

    # Check if courses are in the same curriculum
    for curriculum_id, course_list in curricula.items():
        if course1 in course_list and course2 in course_list:
            return True
        
def get_assigned_courses_by_period(day, period):
    courses = []
    for room in timetable[day][period]:
        if timetable[day][period][room] != -1:
            courses.append(timetable[day][period][room])
    return courses

def get_room(day, period, course):
    for room in timetable[day][period]:
        if timetable[day][period][room] == course:
            return room

def is_complete():
    for course in courses:
        if courses[course]['lectures'] != courses[course]['assigned_lectures']:
            return False
    return True
    
def assign_courses():

    sequenced_courses = sorted(courses.keys(), key=lambda c: -len(unavailability_constraints.get(c, [])))

    for _ in range(20):       
        #**** Procedure 1 *****#
        for course in sequenced_courses:
            for _ in range(courses[course]['lectures'] - courses[course]['assigned_lectures']):
                available_slots = get_available_slots(course)
                if available_slots:
                    slot = available_slots[random.randint(0,len(available_slots)-1)]
                    timetable[slot[0]][slot[1]][slot[2]] = course
                    courses[course]['assigned_lectures'] += 1

        if is_complete(): 
            break 

    
    #**** Procedure 2 *****#
    for course in sequenced_courses:
        unassigned_courses = courses[course]['lectures'] - courses[course]['assigned_lectures']
        for _ in range(unassigned_courses):
            available_slots, conflict_course = get_available_slots_with_conflict_course(course)
            if available_slots:
                rnd = random.randint(0,len(available_slots)-1)
                slot = available_slots[rnd]
                target_course = conflict_course[rnd]
                available_slots.pop(rnd)
                if get_available_slots(target_course, slot):
                    timetable[slot[0]][slot[1]][slot[2]] = course
                    courses[course]['assigned_lectures'] += 1
                    #Reassign
                    target_available_slots = get_available_slots(target_course, slot)
                    target_slot = target_available_slots[random.randint(0,len(target_available_slots)-1)]
                    timetable[slot[0]][slot[1]][get_room(slot[0], slot[1], target_course)] = -1
                    timetable[target_slot[0]][target_slot[1]][target_slot[2]] = target_course

        if is_complete(): break  


    #****** Procedure 3 ********#
    for course in sequenced_courses:
        unassigned_courses = courses[course]['lectures'] - courses[course]['assigned_lectures']
        for _ in range(unassigned_courses):
            available_slots = get_swappable_slots(course)
            if available_slots:
                rnd = random.randint(0,len(available_slots)-1)
                slot = available_slots[rnd]
                target_course = timetable[slot[0]][slot[1]][slot[2]]
                timetable[slot[0]][slot[1]][slot[2]] = course
                courses[course]['assigned_lectures'] += 1
                #Reassign
                target_available_slots = get_available_slots(target_course, slot)
                target_slot = target_available_slots[random.randint(0,len(target_available_slots)-1)]
                timetable[target_slot[0]][target_slot[1]][target_slot[2]] = target_course

        if is_complete(): break  


    #****** Procedure 4 ********#
    for course in sequenced_courses:
        unassigned_courses = courses[course]['lectures'] - courses[course]['assigned_lectures']
        for _ in range(unassigned_courses):
            available_slots, conflict_courses = get_overridable_slots(course)
            if available_slots:
                rnd = random.randint(0,len(available_slots)-1)
                slot = available_slots[rnd]
                conflict_course = conflict_courses[rnd]
                target_course = timetable[slot[0]][slot[1]][slot[2]]
                timetable[slot[0]][slot[1]][slot[2]] = course
                conflict_slot_room = get_room(slot[0], slot[1], conflict_course)
                timetable[slot[0]][slot[1]][conflict_slot_room] = target_course
                target_conflict_available_slots = get_available_slots(conflict_course, slot)
                target_conflict_slot = target_conflict_available_slots[random.randint(0,len(target_conflict_available_slots)-1)]
                timetable[target_conflict_slot[0]][target_conflict_slot[1]][target_conflict_slot[2]] = conflict_course
                courses[course]['assigned_lectures'] += 1
        
        if is_complete(): break

    #******* Swapping Procedure *******#
    #This procedure tries to move all assigned courses to other available slots
    for day in timetable:
        for period in timetable[day]:
            for room in timetable[day][period]:
                course = timetable[day][period][room]
                if timetable[day][period][room] != -1:
                    available_slots = get_available_slots(course, [day, period, room])
                    if available_slots:
                        rnd = random.randint(0,len(available_slots)-1)
                        slot = available_slots[rnd]
                        timetable[day][period][room] = -1
                        timetable[slot[0]][slot[1]][slot[2]] = course
                        

    if is_complete():
        #solutionCount+= 1
        #with open("output/out"+str(solutionCount+1)+".out", "w") as file:  # Open the file in write mode
            #for day in timetable:
                #for period in timetable[day]:
                    #for room in timetable[day][period]:
                        #if timetable[day][period][room] != -1:
                            #file.write(f"{timetable[day][period][room]} {room} {day} {period}\n") 
        return timetable
        #******* Procedure 5 *******#
    reset_timetable()

       
delete_all_files_in_output()  
assign_courses()




