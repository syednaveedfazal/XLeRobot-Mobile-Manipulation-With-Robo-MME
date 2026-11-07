"""Custom manager terms for XLeRobot environments.

Isaac Lab builds environments from "terms": small functions or classes that compute one
observation, apply one part of the action, and so on. These are the two XLeRobot needs
on top of Isaac Lab's built-in ones (imported below as `mdp`).
"""

from dataclasses import MISSING

import torch
from isaaclab.envs.mdp import *  # noqa: F401, F403  (re-export Isaac Lab's built-in terms)
from isaaclab.managers import ActionTerm, ActionTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from xle_leisaac.robots.xlerobot import XLEROBOT_JOINT_NAMES


class BaseVelocityAction(ActionTerm):
    """Drive the base with body-frame speeds, like the real robot's x.vel / y.vel / theta.vel.

    Input (3 values): forward m/s, left m/s, turn rad/s, in the robot's own frame.
    The virtual x and y joints slide along WORLD axes, so the forward/left part is rotated
    by the robot's current heading before it becomes a joint speed.
    """

    cfg: "BaseVelocityActionCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._robot = env.scene[cfg.asset_name]
        self._joint_ids = self._robot.find_joints(cfg.joint_names, preserve_order=True)[0]
        self._raw = torch.zeros(self.num_envs, 3, device=self.device)
        self._processed = torch.zeros_like(self._raw)

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed

    def process_actions(self, actions: torch.Tensor):
        self._raw[:] = actions
        yaw = self._robot.data.joint_pos[:, self._joint_ids[2]]
        forward, left = actions[:, 0], actions[:, 1]
        self._processed[:, 0] = forward * torch.cos(yaw) - left * torch.sin(yaw)
        self._processed[:, 1] = forward * torch.sin(yaw) + left * torch.cos(yaw)
        self._processed[:, 2] = actions[:, 2]

    def apply_actions(self):
        self._robot.set_joint_velocity_target(self._processed, joint_ids=self._joint_ids)

    def reset(self, env_ids=None):
        self._raw[env_ids] = 0.0


@configclass
class BaseVelocityActionCfg(ActionTermCfg):
    class_type: type = BaseVelocityAction
    joint_names: list[str] = MISSING


def xlerobot_state(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The 17 values the real robot reports, in sim units and in action-key order.

    14 joint positions (both arms, then the head) in radians, then the base's body-frame
    speeds: forward m/s, left m/s, turn rad/s. sim_to_lerobot_action() converts it to real units.
    """
    robot = env.scene[asset_cfg.name]
    ids = robot.find_joints(XLEROBOT_JOINT_NAMES, preserve_order=True)[0]
    positions = robot.data.joint_pos[:, ids[:14]]
    x_vel, y_vel, turn = (robot.data.joint_vel[:, i] for i in ids[14:])
    yaw = robot.data.joint_pos[:, ids[16]]
    forward = torch.cos(yaw) * x_vel + torch.sin(yaw) * y_vel
    left = -torch.sin(yaw) * x_vel + torch.cos(yaw) * y_vel
    return torch.cat([positions, torch.stack([forward, left, turn], dim=-1)], dim=-1)

