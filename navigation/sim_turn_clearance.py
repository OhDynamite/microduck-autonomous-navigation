"""Simulation-only turn clearance evaluation.

This module uses MuJoCo geometry information.
It must not be treated as real sensor evidence.
It does not authorize movement or bypass the Phase 4 safety gate.
"""

import math

import mujoco
import numpy as np


def evaluate_sim_turn_clearance(
    model,
    data,
    *,
    clearance_margin=0.15,
    robot_radius=0.10,
):
    """Evaluate conservative XY clearance for stationary rotation.

    Returns:
        (safe, minimum_clearance, reason)

    This is a simulation diagnostic, not movement authorization.
    """

    robot_xy = np.asarray(data.qpos[:2], dtype=float)

    minimum_clearance = float("inf")
    obstacle_count = 0

    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_id,
        )

        if not name or not name.startswith("obstacle"):
            continue

        geom_type = model.geom_type[geom_id]

        if geom_type != mujoco.mjtGeom.mjGEOM_BOX:
            return (
                False,
                None,
                f"Unsupported obstacle geometry: {name}",
            )

        obstacle_count += 1

        center = np.asarray(data.geom_xpos[geom_id][:2], dtype=float)
        rotation = np.asarray(
            data.geom_xmat[geom_id],
            dtype=float,
        ).reshape(3, 3)

        half_sizes = np.asarray(model.geom_size[geom_id][:2], dtype=float)

        # Conservative XY bounding circle.
        obstacle_radius = math.hypot(
            float(half_sizes[0]),
            float(half_sizes[1]),
        )

        center_distance = float(np.linalg.norm(robot_xy - center))

        surface_clearance = (
            center_distance - obstacle_radius - robot_radius
        )

        minimum_clearance = min(
            minimum_clearance,
            surface_clearance,
        )

    if obstacle_count == 0:
        return False, None, "No verified obstacle geometry"

    safe = minimum_clearance >= clearance_margin

    reason = (
        "SIM_CLEARANCE_OK"
        if safe
        else "SIM_CLEARANCE_BLOCKED"
    )

    return safe, minimum_clearance, reason
