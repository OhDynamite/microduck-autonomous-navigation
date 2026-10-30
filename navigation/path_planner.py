"""Simulator-independent A* path planner for MicroDuck."""

import heapq
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Obstacle:
    x: float
    y: float
    width: float
    height: float


def blocked(x, y, obstacles, margin):
    for o in obstacles:
        if (abs(x - o.x) <= o.width / 2 + margin
                and abs(y - o.y) <= o.height / 2 + margin):
            return True
    return False


def segment_clear(a, b, obstacles, margin):
    """Check that a segment does not intersect an inflated obstacle."""
    for o in obstacles:
        xmin = o.x - o.width / 2 - margin
        xmax = o.x + o.width / 2 + margin
        ymin = o.y - o.height / 2 - margin
        ymax = o.y + o.height / 2 + margin

        dx = b[0] - a[0]
        dy = b[1] - a[1]
        tmin, tmax = 0.0, 1.0

        for p, d, low, high in (
            (a[0], dx, xmin, xmax),
            (a[1], dy, ymin, ymax),
        ):
            if abs(d) < 1e-12:
                if low <= p <= high:
                    return False
                continue

            t1 = (low - p) / d
            t2 = (high - p) / d
            tmin = max(tmin, min(t1, t2))
            tmax = min(tmax, max(t1, t2))

        if tmin <= tmax:
            return False

    return True


def simplify_path(path, obstacles, margin):
    if len(path) <= 2:
        return path

    result = [path[0]]
    i = 0

    while i < len(path) - 1:
        j = len(path) - 1

        while j > i + 1:
            if segment_clear(path[i], path[j], obstacles, margin):
                break
            j -= 1

        result.append(path[j])
        i = j

    return result


def plan_path(
    start,
    goal,
    obstacles,
    resolution=0.05,
    margin=0.15,
    bounds=(-0.30, 1.30, -0.40, 1.00),
):
    """Return simplified waypoints including start and goal."""

    if resolution <= 0 or margin < 0:
        raise ValueError("Invalid resolution or margin")

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
        x, y = cell
        if not (0 <= x <= nx and 0 <= y <= ny):
            return False
        return not blocked(*to_world(cell), obstacles, margin)

    start_cell = to_grid(start)
    goal_cell = to_grid(goal)

    if not valid(start_cell) or not valid(goal_cell):
        raise ValueError("Start or goal is outside the free space")

    directions = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    ]

    def heuristic(cell):
        return math.hypot(
            goal_cell[0] - cell[0],
            goal_cell[1] - cell[1],
        )

    queue = [(heuristic(start_cell), 0, start_cell)]
    costs = {start_cell: 0.0}
    parents = {}
    count = 0

    while queue:
        _, _, current = heapq.heappop(queue)

        if current == goal_cell:
            cells = [current]

            while cells[-1] != start_cell:
                cells.append(parents[cells[-1]])

            cells.reverse()
            path = [to_world(cell) for cell in cells]
            path[0] = tuple(start)
            path[-1] = tuple(goal)

            result = simplify_path(path, obstacles, margin)

            if not all(
                segment_clear(a, b, obstacles, margin)
                for a, b in zip(result, result[1:])
            ):
                raise RuntimeError("Path safety validation failed")

            return result

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)

            if not valid(neighbor):
                continue

            if not segment_clear(
                to_world(current),
                to_world(neighbor),
                obstacles,
                margin,
            ):
                continue

            new_cost = costs[current] + math.hypot(dx, dy)

            if new_cost >= costs.get(neighbor, math.inf) - 1e-10:
                continue

            costs[neighbor] = new_cost
            parents[neighbor] = current
            count += 1

            heapq.heappush(
                queue,
                (new_cost + heuristic(neighbor), count, neighbor),
            )

    raise RuntimeError("No collision-free path found")


if __name__ == "__main__":
    obstacles = [Obstacle(0.50, 0.25, 0.20, 0.20)]

    path = plan_path(
        start=(0.00, 0.00),
        goal=(1.00, 0.50),
        obstacles=obstacles,
    )

    print("=== A* Path Planning ===")
    print(f"Generated {len(path)} points")

    for i, (x, y) in enumerate(path):
        print(f"Point {i}: X={x:.3f}, Y={y:.3f}")

    print("PATH PLANNING SUCCESS")
