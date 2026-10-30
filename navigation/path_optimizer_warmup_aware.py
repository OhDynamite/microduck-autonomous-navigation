"""Experimental one-waypoint optimizer using predicted WALK state."""

import math

from path_planner import segment_clear
from path_planner_turn_aware import (
    V_REF,
    W_REF,
    angle_difference,
)


def predict_warmup_state(
    start,
    initial_yaw=0.0,
    warmup_time=2.02,
    warmup_speed=0.10,
):
    if warmup_time < 0 or warmup_speed < 0:
        raise ValueError("Invalid warmup parameters")

    x = start[0] + warmup_speed * warmup_time * math.cos(initial_yaw)
    y = start[1] + warmup_speed * warmup_time * math.sin(initial_yaw)
    return (x, y), initial_yaw


def estimate_cost(
    warmup_position,
    warmup_yaw,
    waypoint,
    goal,
    warmup_time=2.02,
    forward_speed=V_REF,
    turn_speed=W_REF,
):
    if forward_speed <= 0 or turn_speed <= 0:
        raise ValueError("Speeds must be positive")

    points = [warmup_position, waypoint, goal]
    distance = 0.0
    turning = 0.0
    previous_yaw = warmup_yaw

    for a, b in zip(points, points[1:]):
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        length = math.hypot(dx, dy)

        if length <= 1e-12:
            raise ValueError("Zero-length path segment")

        yaw = math.atan2(dy, dx)
        distance += length
        turning += angle_difference(yaw, previous_yaw)
        previous_yaw = yaw

    return (
        warmup_time
        + distance / forward_speed
        + turning / turn_speed
    )


def optimize_warmup_waypoint(
    start,
    goal,
    obstacles,
    margin=0.15,
    resolution=0.05,
    initial_yaw=0.0,
    warmup_time=2.02,
    warmup_speed=0.10,
    min_segment_length=0.20,
    search_padding=0.50,
):
    """Return START -> best safe WP -> GOAL within a local search region."""

    if resolution <= 0 or margin < 0:
        raise ValueError("Invalid planner parameters")
    if min_segment_length < 0 or search_padding < 0:
        raise ValueError("Invalid search parameters")

    warmup_pos, warmup_yaw = predict_warmup_state(
        start,
        initial_yaw=initial_yaw,
        warmup_time=warmup_time,
        warmup_speed=warmup_speed,
    )

    min_x = min(start[0], goal[0]) - search_padding
    max_x = max(start[0], goal[0]) + search_padding
    min_y = min(start[1], goal[1]) - search_padding
    max_y = max(start[1], goal[1]) + search_padding

    ix_min = math.floor(min_x / resolution)
    ix_max = math.ceil(max_x / resolution)
    iy_min = math.floor(min_y / resolution)
    iy_max = math.ceil(max_y / resolution)

    candidates = []

    for ix in range(ix_min, ix_max + 1):
        for iy in range(iy_min, iy_max + 1):
            wp = (
                round(ix * resolution, 10),
                round(iy * resolution, 10),
            )

            if math.dist(start, wp) <= min_segment_length:
                continue
            if math.dist(warmup_pos, wp) <= min_segment_length:
                continue
            if math.dist(wp, goal) <= min_segment_length:
                continue

            if not segment_clear(start, wp, obstacles, margin):
                continue
            if not segment_clear(warmup_pos, wp, obstacles, margin):
                continue
            if not segment_clear(wp, goal, obstacles, margin):
                continue

            cost = estimate_cost(
                warmup_pos,
                warmup_yaw,
                wp,
                goal,
                warmup_time=warmup_time,
            )

            candidates.append((cost, wp))

    if not candidates:
        raise RuntimeError(
            "No valid one-waypoint path found. "
            "Multi-waypoint planning may be required."
        )

    candidates.sort()
    best_cost, best_wp = candidates[0]

    return {
        "path": [tuple(start), best_wp, tuple(goal)],
        "predicted_warmup_position": warmup_pos,
        "predicted_cost": best_cost,
        "candidate_count": len(candidates),
    }
