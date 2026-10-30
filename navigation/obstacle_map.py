
import numpy as np


class ObstacleMap:
    """Stores detected obstacle surface points in world XY."""

    def __init__(self):
        self.points = []

    def register(self, origin, direction, distance):
        if distance is None:
            return None

        origin = np.asarray(origin, dtype=float)
        direction = np.asarray(direction, dtype=float)

        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            raise ValueError("Invalid sensor direction")

        direction = direction / norm

        hit = origin + direction * float(distance)

        point = (float(hit[0]), float(hit[1]))
        self.points.append(point)

        return point
