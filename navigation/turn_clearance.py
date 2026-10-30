import math
import numbers

LEFT_RADIUS = 0.13984
RIGHT_RADIUS = 0.14741
DEFAULT_MARGIN = 0.05


def required_clearance(direction, margin=DEFAULT_MARGIN):
    if direction == "LEFT":
        radius = LEFT_RADIUS
    elif direction == "RIGHT":
        radius = RIGHT_RADIUS
    else:
        raise ValueError("Invalid direction")

    if not math.isfinite(margin) or margin < 0:
        raise ValueError("Invalid safety margin")

    return radius + margin


def evaluate_turn_clearance(direction, obstacle_distance, margin=DEFAULT_MARGIN):
    """Conservative circular clearance check for a surface distance.

    obstacle_distance must be the nearest horizontal distance from
    the turn-start position to an obstacle surface, not raw ToF range.
    This check is preliminary and does not certify collision safety.
    """
    required = required_clearance(direction, margin)

    if isinstance(obstacle_distance, bool):
        return False

    if not isinstance(obstacle_distance, numbers.Real):
        return False

    if not math.isfinite(obstacle_distance) or obstacle_distance < 0:
        return False

    return obstacle_distance > required
