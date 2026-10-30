"""Simulation-only conservative turn drift clearance check.

This is a preliminary geometric test, not a guarantee of
physical safety or real-robot authorization.
"""

import math


def evaluate_turn_drift_clearance(
    position,
    obstacles,
    *,
    margin=0.15,
    drift_budget=0.08,
):
    x, y = map(float, position)

    if not all(math.isfinite(v) for v in (
        x, y, margin, drift_budget
    )):
        return False, 0.0, "INVALID_INPUT"

    if margin < 0 or drift_budget < 0:
        return False, 0.0, "INVALID_MARGIN"

    if not obstacles:
        return False, 0.0, "NO_OBSTACLE_INFORMATION"

    minimum_clearance = math.inf

    for obstacle in obstacles:
        xmin = obstacle.x - obstacle.width / 2 - margin
        xmax = obstacle.x + obstacle.width / 2 + margin
        ymin = obstacle.y - obstacle.height / 2 - margin
        ymax = obstacle.y + obstacle.height / 2 + margin

        if xmin <= x <= xmax and ymin <= y <= ymax:
            return False, 0.0, "INSIDE_EXCLUSION_ZONE"

        dx = max(xmin - x, 0.0, x - xmax)
        dy = max(ymin - y, 0.0, y - ymax)

        clearance = math.hypot(dx, dy)
        minimum_clearance = min(minimum_clearance, clearance)

    if (
        minimum_clearance <= drift_budget
        or math.isclose(
            minimum_clearance,
            drift_budget,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    ):
        return False, minimum_clearance, "INSUFFICIENT_DRIFT_CLEARANCE"

    return True, minimum_clearance, "DRIFT_CLEARANCE_OK"
