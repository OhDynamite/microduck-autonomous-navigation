#!/usr/bin/env python3
"""MicroDuck scenario configuration and validation."""

import json
import math
from pathlib import Path

import mujoco

from path_planner import Obstacle


def load_scenario(path):
    """Load scenario settings from JSON."""
    config = json.loads(
        Path(path).read_text(encoding="utf-8")
    )

    if not config.get("obstacles"):
        raise ValueError("Scenario has no obstacles.")

    names = set()

    for obs in config["obstacles"]:
        name = str(obs["name"])

        if not name or name in names:
            raise ValueError(
                f"Invalid or duplicate obstacle name: {name}"
            )

        names.add(name)

        for key in ("x", "y", "z", "width", "depth", "height"):
            value = float(obs[key])
            if not math.isfinite(value):
                raise ValueError(
                    f"{name}: {key} must be finite"
                )

        for key in ("width", "depth", "height"):
            if float(obs[key]) <= 0:
                raise ValueError(
                    f"{name}: {key} must be positive"
                )

    return config


def validate_scene(model, config, tolerance=1e-6):
    """Check whether MuJoCo obstacles match JSON settings."""

    errors = []

    for obs in config["obstacles"]:
        name = str(obs["name"])

        geom_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            name,
        )

        if geom_id < 0:
            errors.append(f"{name}: geom not found")
            continue

        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
            errors.append(f"{name}: geom is not a box")
            continue

        expected_pos = (
            float(obs["x"]),
            float(obs["y"]),
            float(obs["z"]),
        )

        expected_size = (
            float(obs["width"]) / 2,
            float(obs["depth"]) / 2,
            float(obs["height"]) / 2,
        )

        actual_pos = model.geom_pos[geom_id]
        actual_size = model.geom_size[geom_id]

        for i in range(3):
            if abs(actual_pos[i] - expected_pos[i]) > tolerance:
                errors.append(
                    f"{name}: position mismatch axis {i}"
                )

            if abs(actual_size[i] - expected_size[i]) > tolerance:
                errors.append(
                    f"{name}: size mismatch axis {i}"
                )

    if errors:
        raise ValueError(
            "Scenario/MuJoCo mismatch:\n"
            + "\n".join(errors)
        )

    return True


def planning_obstacles_from_config(config):
    """Convert scenario obstacles to A* obstacle objects."""
    return [
        Obstacle(
            float(obs["x"]),
            float(obs["y"]),
            float(obs["width"]),
            float(obs["depth"]),
        )
        for obs in config["obstacles"]
    ]
