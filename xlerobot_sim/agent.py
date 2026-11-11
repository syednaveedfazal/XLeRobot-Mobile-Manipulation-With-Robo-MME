"""XLeRobot agent: the vendored upstream ManiSkill agent, corrected to match the real robot.

The vendored files in third_party/ stay unmodified; the fixes live in this subclass, which is
registered under the same uid ("xlerobot"):

  * mass      the upstream URDF gives base_link 70.1 kg and the head 3.2 kg (robot total 79 kg).
              The real robot is ~12 kg (XLeRobot hardware docs). Body links are set to the masses
              of our Isaac model (base 5.85, top base 4.35, head pan 0.25, head tilt 0.20 kg), so
              the total is ~12.3 kg. Inertia is scaled with the mass.
  * cameras   640x480 like the real cameras (docs record at 640x480). Field of view as in our
              Isaac model: head 69 deg horizontal (RealSense D415 colour), wrists 120 deg horizontal.
  * control   new mode ``pd_joint_pos_real_layout``: 17-dim action in the real robot's key order,
              absolute joint targets, holonomic base (x, y, yaw velocity in the robot frame)
              clamped to the real robot's fastest preset (0.3 m/s, 90 deg/s). See `layout.py`.
              The upstream modes (e.g. 16-dim ``pd_joint_delta_pos_dual_arm``) still work.
  * obs       ``agent/state_real_layout``: 17-dim state in the same order as the action
              (joint positions, then base velocity in the robot frame).
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch

from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.controllers.pd_base_vel import PDBaseVelController, PDBaseVelControllerConfig
from mani_skill.agents.registration import register_agent

from xlerobot_sim import layout

REPO_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_DIR = REPO_ROOT / "third_party" / "xlerobot_maniskill"
XLEROBOT_URDF = UPSTREAM_DIR / "assets" / "xlerobot" / "xlerobot.urdf"


def _load_upstream_agent():
    """Import the vendored agent module (its @register_agent registers uid 'xlerobot')."""
    name = "xlerobot_sim._upstream_xlerobot_agent"
    if name in sys.modules:
        return sys.modules[name]
    path = UPSTREAM_DIR / "agents" / "xlerobot" / "xlerobot.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    # upstream expects the URDF inside mani_skill's package assets; point it at our copy instead
    module.Xlerobot.urdf_path = str(XLEROBOT_URDF)
    return module


UpstreamXlerobot = _load_upstream_agent().Xlerobot


def _vfov(hfov_deg: float, width: int, height: int) -> float:
    """Vertical field of view (rad) for a horizontal one (deg); ManiSkill's CameraConfig.fov is vertical."""
    return 2.0 * math.atan(math.tan(math.radians(hfov_deg) / 2.0) * height / width)


class ClippedPDBaseVelController(PDBaseVelController):
    """Holonomic base velocity (robot frame), clamped to the action-space bounds."""

    def _preprocess_action(self, action):
        action = super()._preprocess_action(action)
        low = torch.as_tensor(self.single_action_space.low, device=action.device, dtype=action.dtype)
        high = torch.as_tensor(self.single_action_space.high, device=action.device, dtype=action.dtype)
        return torch.clamp(action, low, high)


class ClippedPDBaseVelControllerConfig(PDBaseVelControllerConfig):
    controller_cls = ClippedPDBaseVelController


@register_agent(override=True)
class XLeRobot(UpstreamXlerobot):
    uid = "xlerobot"

    BODY_MASSES = {"base_link": 5.8538, "top_base_link": 4.3542, "head_pan_link": 0.25, "head_tilt_link": 0.20}
    CAMERA_SIZE = (640, 480)
    CAMERA_HFOV_DEG = {"fetch_head": 69.0, "fetch_right_arm_camera": 120.0, "fetch_left_arm_camera": 120.0}

    @property
    def _sensor_configs(self):
        configs = super()._sensor_configs
        w, h = self.CAMERA_SIZE
        for c in configs:
            c.width, c.height = w, h
            c.fov = _vfov(self.CAMERA_HFOV_DEG[c.uid], w, h)
        return configs

    def _after_loading_articulation(self):
        super()._after_loading_articulation()
        for link in self.robot.get_links():
            mass = self.BODY_MASSES.get(link.name)
            if mass is None:
                continue
            for body in link._objs:  # one physx link per parallel env
                scale = mass / body.mass
                body.mass = mass
                body.inertia = np.asarray(body.inertia) * scale

    @property
    def _controller_configs(self):
        configs = super()._controller_configs

        def joint_pos(names, stiffness, damping, force_limit):
            # absolute targets in radians; bounds = URDF joint limits
            return PDJointPosControllerConfig(list(names), None, None, stiffness, damping, force_limit,
                                              normalize_action=False)

        configs[layout.REAL_LAYOUT_MODE] = dict(  # dict order = action order (see layout.py)
            left_arm=joint_pos(layout.SIM_LEFT_ARM[:5], self.arm2_stiffness, self.arm2_damping,
                               self.arm2_force_limit),
            left_gripper=joint_pos(layout.SIM_LEFT_ARM[5:], self.gripper2_stiffness, self.gripper2_damping,
                                   self.gripper2_force_limit),
            right_arm=joint_pos(layout.SIM_RIGHT_ARM[:5], self.arm_stiffness, self.arm_damping,
                                self.arm_force_limit),
            right_gripper=joint_pos(layout.SIM_RIGHT_ARM[5:], self.gripper_stiffness, self.gripper_damping,
                                    self.gripper_force_limit),
            head=joint_pos(layout.SIM_HEAD, self.body_stiffness, self.body_damping, self.body_force_limit),
            base=ClippedPDBaseVelControllerConfig(
                list(layout.SIM_BASE),
                lower=[-layout.BASE_MAX_XY, -layout.BASE_MAX_XY, -layout.BASE_MAX_YAW],
                upper=[layout.BASE_MAX_XY, layout.BASE_MAX_XY, layout.BASE_MAX_YAW],
                damping=1000,
                force_limit=500,
                normalize_action=False,
            ),
        )
        return configs

    def _after_init(self):
        super()._after_init()
        names = [j.name for j in self.robot.get_active_joints()]
        self._layout_pos_idx = torch.tensor([names.index(n) for n in layout.SIM_POSITION_JOINTS], device=self.device)
        self._layout_base_idx = [names.index(n) for n in layout.SIM_BASE]

    def real_layout_state(self) -> torch.Tensor:
        """(num_envs, 17): joint positions (rad) then base x/y velocity (m/s, robot frame) and yaw rate (rad/s)."""
        qpos, qvel = self.robot.get_qpos(), self.robot.get_qvel()
        bx, by, byaw = self._layout_base_idx
        yaw = qpos[:, byaw]
        c, s = torch.cos(yaw), torch.sin(yaw)
        vx_w, vy_w = qvel[:, bx], qvel[:, by]
        base = torch.stack([c * vx_w + s * vy_w, -s * vx_w + c * vy_w, qvel[:, byaw]], dim=-1)
        return torch.cat([qpos[:, self._layout_pos_idx], base], dim=-1)

    def base_pose(self) -> torch.Tensor:
        """(num_envs, 3): base x, y (m) and yaw (rad) from the virtual base joints."""
        return self.robot.get_qpos()[:, self._layout_base_idx]

    def get_proprioception(self):
        obs = super().get_proprioception()
        obs["state_real_layout"] = self.real_layout_state()
        return obs
