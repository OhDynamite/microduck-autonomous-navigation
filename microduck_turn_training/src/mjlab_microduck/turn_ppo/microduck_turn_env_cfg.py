"""MicroDuck turn-in-place PPO environment.

Derived from the official velocity environment configuration.
The official source files are not modified.
"""

from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)


def make_microduck_turn_env_cfg(play: bool = False):
    """Create a turn-focused PPO environment configuration."""

    cfg = make_microduck_velocity_env_cfg(
        play=play,
        rough=False,
    )

    command = cfg.commands["twist"]

    # Train exclusively on turn-in-place commands.
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.rel_turn_in_place_envs = 1.0

    # No commanded linear movement.
    command.ranges.lin_vel_x = (0.0, 0.0)
    command.ranges.lin_vel_y = (0.0, 0.0)

    # Both left and right turns.
    command.ranges.ang_vel_z = (-1.0, 1.0)

    return cfg
