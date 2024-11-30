from constraints import HardConstraints
import random


class Particle:
    def __init__(self, initial_schedule):
        self.schedule = initial_schedule
        self.fitness = float("inf")
        self.best = None
        self.best_fitness = float("inf")
        self.velocity = []

    def update(self, data, personal_best, global_best, chi, c1, c2):
        """
        Updates the velocity and position of the particle.
        Uses the multi-swarm PSO update formula.
        """
        velocity = []

        for i, entry in enumerate(self.schedule):
            r1, r2 = random.random(), random.random()

            # Personal and global best positions
            best_entry = personal_best.schedule[i] if personal_best else entry
            global_best_entry = global_best.schedule[i] if global_best else entry

            # Velocity calculation using PSO formula
            velocity_day = (
                chi * (c1 * r1 * (best_entry["day"] - entry["day"]) +
                       c2 * r2 * (global_best_entry["day"] - entry["day"]))
            )
            velocity_period = (
                chi * (c1 * r1 * (best_entry["period"] - entry["period"]) +
                       c2 * r2 * (global_best_entry["period"] - entry["period"]))
            )
            new_day = max(0, int(entry["day"] + velocity_day))
            new_period = max(0, int(entry["period"] + velocity_period))
            new_room = entry["room_id"]

            # Ensure feasibility before applying the change
            if HardConstraints.check_room_occupancy(data["timetable"], {"day": new_day, "period": new_period, "room_id": new_room, "course_id": entry["course_id"]}):
                continue

            # Append the calculated velocity as an action
            velocity.append({
                "action": "reassign",
                "event": entry["course_id"],
                "new_day": new_day,
                "new_period": new_period,
                "new_room": new_room
            })

        # Apply the velocity actions
        for action in velocity:
            for event in self.schedule:
                if event["course_id"] == action["event"]:
                    event["day"] = action["new_day"]
                    event["period"] = action["new_period"]
                    event["room_id"] = action["new_room"]
                    break

        self.velocity = velocity
