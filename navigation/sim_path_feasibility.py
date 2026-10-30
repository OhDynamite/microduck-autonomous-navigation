"""Simulation-only path feasibility evaluation.

Combines path segment geometry and a conservative turn
displacement budget. Does not authorize robot movement.
"""

import math

from path_planner_safety_fixed import segment_clear
from sim_turn_drift_safety import evaluate_turn_drift_clearance


def evaluate_path_feasibility(
    path,
    obstacles,
    *,
    margin=0.15,
    turn_drift_budget=0.08,
    initial_yaw=0.0,
    turn_threshold_deg=8.0,
):
    if path is None or len(path) < 2:
        return False, "INVALID_PATH"

    if not obstacles:
        return False, "NO_OBSTACLE_INFORMATION"

    for a, b in zip(path, path[1:]):
        if not segment_clear(a, b, obstacles, margin):
            return False, "UNSAFE_SEGMENT"

    current_yaw = float(initial_yaw)

    for index, (a, b) in enumerate(zip(path, path[1:])):
        heading = math.atan2(b[1] - a[1], b[0] - a[0])

        error = math.atan2(
            math.sin(heading - current_yaw),
            math.cos(heading - current_yaw),
        )

        if abs(math.degrees(error)) > turn_threshold_deg:
            safe, clearance, reason = evaluate_turn_drift_clearance(
                a,
                obstacles,
                margin=margin,
                drift_budget=turn_drift_budget,
            )

            if not safe:
                return False, f"TURN_{index}_{reason}"

        current_yaw = heading

    return True, "PATH_FEASIBLE"
