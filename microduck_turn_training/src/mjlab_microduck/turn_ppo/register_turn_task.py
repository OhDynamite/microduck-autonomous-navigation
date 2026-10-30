"""Register the isolated MicroDuck turn PPO task.

No official repository files are modified.
"""

from copy import deepcopy

from mjlab.tasks.registry import register_mjlab_task

from mjlab_microduck.tasks import (
    MicroduckRlCfg,
    MicroduckOnPolicyRunner,
)

from .microduck_turn_train_env_cfg import (
    make_microduck_turn_train_env_cfg,
)


TASK_ID = "Mjlab-Turn-Flat-MicroDuck"


def register_turn_task():
    """Register the turn-focused PPO training task."""

    train_cfg = make_microduck_turn_train_env_cfg(play=False)
    play_cfg = make_microduck_turn_train_env_cfg(play=True)

    rl_cfg = deepcopy(MicroduckRlCfg)

    rl_cfg.experiment_name = "turn_ppo"
    rl_cfg.run_name = "turn_phase5"

    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=train_cfg,
        play_env_cfg=play_cfg,
        rl_cfg=rl_cfg,
        runner_cls=MicroduckOnPolicyRunner,
    )

    return TASK_ID
