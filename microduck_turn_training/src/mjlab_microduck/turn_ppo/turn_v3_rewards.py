"""V3 rewards for directional turn-in-place PPO.

Reward terms are training-only.
No additional Actor observations are introduced.
"""

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import SceneEntityCfg


def reverse_turn_penalty(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    target_wz = command[:, 2]
    actual_wz = robot.data.root_link_ang_vel_b[:, 2]

    return torch.clamp(
        -(target_wz * actual_wz),
        min=0.0,
        max=2.0,
    )


def turn_xy_drift_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]

    position = robot.data.root_link_pos_w[:, :2]
    origin = env.scene.env_origins[:, :2]

    drift_squared = torch.sum(
        torch.square(position - origin),
        dim=1,
    )

    return torch.clamp(drift_squared, max=0.25)
