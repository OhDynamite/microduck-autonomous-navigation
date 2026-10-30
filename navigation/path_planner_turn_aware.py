"""Turn-aware A* planner for MicroDuck (under development)."""

import math
import heapq

V_REF = 0.11
W_REF = 0.18

DIRECTIONS = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
]


def direction_yaw(direction_index):
    """Return heading angle for one of the eight grid directions."""
    dx, dy = DIRECTIONS[direction_index]
    return math.atan2(dy, dx)


def angle_difference(a, b):
    """Return the absolute shortest angular difference in radians."""
    return abs(
        math.atan2(
            math.sin(a - b),
            math.cos(a - b),
        )
    )


def transition_cost(
    previous_yaw,
    direction_index,
    resolution=0.05,
    forward_speed=V_REF,
    turn_speed=W_REF,
):
    """Estimate move + rotation time for a grid transition."""

    if resolution <= 0:
        raise ValueError("resolution must be positive")

    if forward_speed <= 0 or turn_speed <= 0:
        raise ValueError("speeds must be positive")

    dx, dy = DIRECTIONS[direction_index]

    distance = math.hypot(dx, dy) * resolution
    next_yaw = direction_yaw(direction_index)

    turn_angle = angle_difference(next_yaw, previous_yaw)

    move_time = distance / forward_speed
    turn_time = turn_angle / turn_speed

    return move_time + turn_time


def plan_path_turn_aware(
    start,
    goal,
    obstacles,
    initial_yaw=0.0,
    resolution=0.05,
    margin=0.15,
    bounds=(-0.30, 1.30, -0.40, 1.00),
    forward_speed=V_REF,
    turn_speed=W_REF,
):
    """Plan a collision-free grid path with heading-dependent costs."""

    from path_planner import blocked, segment_clear

    if resolution <= 0 or margin < 0:
        raise ValueError("Invalid resolution or margin")

    if forward_speed <= 0 or turn_speed <= 0:
        raise ValueError("Speeds must be positive")

    xmin, xmax, ymin, ymax = bounds

    def to_grid(point):
        return (
            round((point[0] - xmin) / resolution),
            round((point[1] - ymin) / resolution),
        )

    def to_world(cell):
        return (
            round(xmin + cell[0] * resolution, 6),
            round(ymin + cell[1] * resolution, 6),
        )

    nx = math.floor((xmax - xmin) / resolution)
    ny = math.floor((ymax - ymin) / resolution)

    def valid(cell):
        gx, gy = cell
        if not (0 <= gx <= nx and 0 <= gy <= ny):
            return False
        return not blocked(*to_world(cell), obstacles, margin)

    start_cell = to_grid(start)
    goal_cell = to_grid(goal)

    if not valid(start_cell) or not valid(goal_cell):
        raise ValueError("Start or goal is outside free space")

    # Direction index -1 represents the actual initial heading.
    start_state = (*start_cell, -1)

    def heuristic(state):
        dx = (goal_cell[0] - state[0]) * resolution
        dy = (goal_cell[1] - state[1]) * resolution
        return math.hypot(dx, dy) / forward_speed

    queue = [(heuristic(start_state), 0, start_state)]
    costs = {start_state: 0.0}
    parents = {}
    counter = 0

    while queue:
        priority, _, current = heapq.heappop(queue)

        current_cost = costs[current]

        if priority > current_cost + heuristic(current) + 1e-9:
            continue

        if current[:2] == goal_cell:
            states = [current]

            while states[-1] != start_state:
                states.append(parents[states[-1]])

            states.reverse()

            path = [to_world(state[:2]) for state in states]
            path[0] = tuple(start)
            path[-1] = tuple(goal)

            if not all(
                segment_clear(a, b, obstacles, margin)
                for a, b in zip(path, path[1:])
            ):
                raise RuntimeError("Path safety validation failed")

            return path

        previous_yaw = (
            initial_yaw
            if current[2] == -1
            else direction_yaw(current[2])
        )

        for direction_index, (dx, dy) in enumerate(DIRECTIONS):
            neighbor_cell = (current[0] + dx, current[1] + dy)

            if not valid(neighbor_cell):
                continue

            if not segment_clear(
                to_world(current[:2]),
                to_world(neighbor_cell),
                obstacles,
                margin,
            ):
                continue

            next_state = (*neighbor_cell, direction_index)

            edge_cost = transition_cost(
                previous_yaw,
                direction_index,
                resolution=resolution,
                forward_speed=forward_speed,
                turn_speed=turn_speed,
            )

            new_cost = current_cost + edge_cost

            if new_cost >= costs.get(next_state, math.inf) - 1e-10:
                continue

            costs[next_state] = new_cost
            parents[next_state] = current

            counter += 1

            heapq.heappush(
                queue,
                (
                    new_cost + heuristic(next_state),
                    counter,
                    next_state,
                ),
            )

    raise RuntimeError("No collision-free path found")

