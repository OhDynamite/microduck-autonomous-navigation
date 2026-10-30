"""Register an isolated MicroDuck turn transfer PPO V2 task.

V1 registration and pretrained checkpoints remain unchanged.
"""

from copy import deepcopy

from mjlab.tasks.registry import register_mjlab_task
from mjlab_microduck.tasks import MicroduckRlCfg

from .microduck_turn_train_env_cfg_v3 import (
    make_microduck_turn_train_env_cfg_v3,
)
from .transfer_runner import MicroduckTurnTransferRunner


TRANSFER_V3_TASK_ID = "Mjlab-Turn-Transfer-V3-MicroDuck"


def register_transfer_v3_task():
    train_cfg = make_microduck_turn_train_env_cfg_v3(play=False)
    play_cfg = make_microduck_turn_train_env_cfg_v3(play=True)

    rl_cfg = deepcopy(MicroduckRlCfg)
    rl_cfg.experiment_name = "turn_transfer_ppo_v3"
    rl_cfg.run_name = "turn_transfer_phase5_v3"

    register_mjlab_task(
        task_id=TRANSFER_V3_TASK_ID,
        env_cfg=train_cfg,
        play_env_cfg=play_cfg,
        rl_cfg=rl_cfg,
        runner_cls=MicroduckTurnTransferRunner,
    )

    return TRANSFER_V3_TASK_ID
