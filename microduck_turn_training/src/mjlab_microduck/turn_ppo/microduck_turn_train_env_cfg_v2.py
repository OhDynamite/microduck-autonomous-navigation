"""MicroDuck Phase 5 transfer PPO V2 training configuration.

Isolated experimental configuration.
The original training configuration is not modified.
"""

from .microduck_turn_train_env_cfg import (
    make_microduck_turn_train_env_cfg,
)


def make_microduck_turn_train_env_cfg_v2(play: bool = False):
    cfg = make_microduck_turn_train_env_cfg(play=play)

    # Experiment V2: increase yaw tracking emphasis only.
    cfg.rewards["track_angular_velocity"].weight = 6.0

    return cfg
