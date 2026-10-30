"""Turn-aware waypoint optimization for an ordered candidate path."""

import heapq
import math

from path_planner import segment_clear
from path_planner_turn_aware import V_REF, W_REF, angle_difference


def optimize_path(
    path,
    obstacles,
    margin=0.15,
    initial_yaw=0.0,
    forward_speed=V_REF,
    turn_speed=W_REF,
):
    """Find minimum-cost safe shortcuts through ordered path vertices."""

    if len(path) < 2:
        raise ValueError("Path must contain START and GOAL")

    if forward_speed <= 0 or turn_speed <= 0 or margin < 0:
        raise ValueError("Invalid optimization parameters")

    points = [tuple(map(float, p)) for p in path]
    n = len(points)

    # State: (current point index, previous point index)
    # Previous index -1 means the initial robot heading.
    start_state = (0, -1)

    costs = {start_state: 0.0}
    parents = {}
    queue = [(0.0, 0, start_state)]
    counter = 0

    while queue:
        cost, _, state = heapq.heappop(queue)

        if cost > costs[state] + 1e-9:
            continue

        current_idx, previous_idx = state

        if current_idx == n - 1:
            indices = [current_idx]

            while state != start_state:
                state = parents[state]
                indices.append(state[0])

            indices.reverse()
            return [points[i] for i in indices]

        if previous_idx == -1:
            previous_yaw = initial_yaw
        else:
            previous_point = points[previous_idx]
            current_point = points[current_idx]

            previous_yaw = math.atan2(
                current_point[1] - previous_point[1],
                current_point[0] - previous_point[0],
            )

        current_point = points[current_idx]

        for next_idx in range(current_idx + 1, n):
            next_point = points[next_idx]

            if not segment_clear(
                current_point,
                next_point,
                obstacles,
                margin,
            ):
                continue

            dx = next_point[0] - current_point[0]
            dy = next_point[1] - current_point[1]

            distance = math.hypot(dx, dy)

            if distance < 1e-12:
                continue

            next_yaw = math.atan2(dy, dx)

            turn_angle = angle_difference(
                next_yaw,
                previous_yaw,
            )

            edge_cost = (
                distance / forward_speed
                + turn_angle / turn_speed
            )

            new_cost = cost + edge_cost
            next_state = (next_idx, current_idx)

            if new_cost >= costs.get(next_state, math.inf) - 1e-10:
                continue

            costs[next_state] = new_cost
            parents[next_state] = state
            counter += 1

            heapq.heappush(
                queue,
                (new_cost, counter, next_state),
            )

    raise RuntimeError("No collision-free optimized path found")
