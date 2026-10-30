"""MicroDuck turn-focused PPO training configuration.

Uses the isolated turn environment defined in Step 472.
Official files and existing successful policies remain unchanged.
"""

from .microduck_turn_env_cfg import make_microduck_turn_env_cfg


def make_microduck_turn_train_env_cfg(play: bool = False):
    cfg = make_microduck_turn_env_cfg(play=play)

    # Reward turning around the vertical axis.
    cfg.rewards["track_angular_velocity"].weight = 4.0

    # Penalize translational drift during turning.
    cfg.rewards["track_linear_velocity"].weight = 2.0

    # Avoid penalizing the yaw rotation we want to learn.
    cfg.rewards["angular_momentum"].weight = 0.0

    # Permit the leg movements necessary for turning.
    cfg.rewards["pose"].weight = 0.5

    # Maintain upright stability.
    cfg.rewards["upright"].weight = 2.0

    return cfg
