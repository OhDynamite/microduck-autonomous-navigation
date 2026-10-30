import math

from path_planner import Obstacle
from path_planner_turn_aware import plan_path_turn_aware
from path_optimizer_turn_aware import optimize_path
from path_planner_safety_fixed import segment_clear
from safety_map import SafetyMap

import path_planner as base_planner
import path_optimizer_turn_aware as turn_optimizer

# Always use the corrected safety implementation.
base_planner.segment_clear = segment_clear
turn_optimizer.segment_clear = segment_clear

BOUNDS = (-0.30, 1.80, -0.60, 1.00)
RESOLUTION = 0.05
MARGIN = 0.15


def convert_obstacles(safety_map):
    return [
        Obstacle(x, y, 2 * r, 2 * r)
        for x, y, r in safety_map.zones
    ]


def make_path(start, goal, yaw, obstacles):
    grid_path = plan_path_turn_aware(
        start=start,
        goal=goal,
        obstacles=obstacles,
        initial_yaw=yaw,
        resolution=RESOLUTION,
        margin=MARGIN,
        bounds=BOUNDS,
    )

    path = optimize_path(
        grid_path,
        obstacles,
        margin=MARGIN,
        initial_yaw=yaw,
    )

    if not path or len(path) < 2:
        raise RuntimeError("No valid path generated")

    for a, b in zip(path, path[1:]):
        if not segment_clear(a, b, obstacles, MARGIN):
            raise RuntimeError(f"Unsafe segment: {a} -> {b}")

    return path


def main():
    print("===== Step 118: Dynamic Replanning Test =====")

    start = (0.0, 0.0)
    goal = (1.5, 0.0)

    # Initially, no obstacles are known.
    known_obstacles = []

    initial_path = make_path(
        start, goal, 0.0, known_obstacles
    )

    print(f"[INITIAL] Known obstacles: {len(known_obstacles)}")
    print(f"[INITIAL] Path points: {len(initial_path)}")

    assert len(initial_path) >= 2
    print("[OK] Initial path generated")

    # Simulated current robot pose during navigation.
    current_pos = (0.30, 0.0)
    current_yaw = 0.0

    # New ToF detection, using the surface point
    # verified in Step 117.
    detected_point = (0.90, 0.0)

    safety_map = SafetyMap(safety_radius=0.20)
    safety_map.add_detection(detected_point)

    known_obstacles = convert_obstacles(safety_map)

    print(f"[DETECT] New obstacle at: {detected_point}")
    print(f"[DETECT] Known obstacles: {len(known_obstacles)}")

    assert not segment_clear(
        current_pos, goal, known_obstacles, MARGIN
    )
    print("[OK] Existing direct path is blocked")

    new_path = make_path(
        current_pos,
        goal,
        current_yaw,
        known_obstacles,
    )

    print(f"[REPLAN] New path points: {len(new_path)}")

    for i, (x, y) in enumerate(new_path):
        print(f"[PATH] {i}: ({x:.3f}, {y:.3f})")

    assert math.dist(new_path[0], current_pos) < 1e-6
    assert math.dist(new_path[-1], goal) < 1e-6

    print("[OK] Replanning starts at current position")
    print("[OK] Replanned path reaches GOAL")
    print("[OK] All replanned segments safe")

    print()
    print("SUCCESS: Dynamic replanning verified")
    print("===== Step 118 Completed =====")


if __name__ == "__main__":
    main()
