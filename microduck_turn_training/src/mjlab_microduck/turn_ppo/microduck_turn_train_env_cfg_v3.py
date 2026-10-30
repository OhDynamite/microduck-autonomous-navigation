"""MicroDuck turn-focused PPO V3 training configuration."""

from mjlab.managers import RewardTermCfg

from .microduck_turn_train_env_cfg_v2 import (
    make_microduck_turn_train_env_cfg_v2,
)
from .turn_v3_rewards import (
    reverse_turn_penalty,
    turn_xy_drift_penalty,
)


def make_microduck_turn_train_env_cfg_v3(play: bool = False):
    cfg = make_microduck_turn_train_env_cfg_v2(play=play)

    cfg.rewards["reverse_turn"] = RewardTermCfg(
        func=reverse_turn_penalty,
        weight=-4.0,
        params={"command_name": "twist"},
    )

    cfg.rewards["turn_xy_drift"] = RewardTermCfg(
        func=turn_xy_drift_penalty,
        weight=-5.0,
    )

    return cfg
