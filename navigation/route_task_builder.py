"""Step 1278: Pure route-to-control-task conversion.

No MuJoCo access, motor commands or movement authorization.
"""
import math


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def build_route_tasks(points, initial_yaw):
    if len(points) < 2:
        raise ValueError("At least two route points required")

    if not math.isfinite(float(initial_yaw)):
        raise ValueError("Invalid initial yaw")

    cleaned = []
    for point in points:
        if len(point) != 2:
            raise ValueError("Invalid XY point")
        xy = tuple(float(v) for v in point)
        if not all(math.isfinite(v) for v in xy):
            raise ValueError("Nonfinite XY point")
        cleaned.append(xy)

    previous_yaw = float(initial_yaw)
    tasks = []

    for index, (start, goal) in enumerate(
        zip(cleaned, cleaned[1:]), 1
    ):
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        distance = math.hypot(dx, dy)

        if distance < 1e-6:
            raise ValueError("Zero-length route segment")

        target_yaw = math.atan2(dy, dx)
        delta = normalize_angle(target_yaw - previous_yaw)

        tasks.append({
            "segment": index,
            "start_xy": list(start),
            "target_xy": list(goal),
            "length_m": distance,
            "target_yaw_rad": target_yaw,
            "target_yaw_deg": math.degrees(target_yaw),
            "turn_delta_deg": math.degrees(delta),
            "nominal_turn_seconds_at_0_5rad_s":
                abs(delta) / 0.5,
            "task_type": "TURN_THEN_WALK",
            "execution_state": "NOT_STARTED",
            "movement_authorized": False,
        })

        previous_yaw = target_yaw

    return tasks
