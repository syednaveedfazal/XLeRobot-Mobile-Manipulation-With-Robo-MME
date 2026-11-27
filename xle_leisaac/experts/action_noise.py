"""action_noise.py - recovery demonstrations (DART): execute noisy actions, record clean ones.

A policy trained only on perfect demos never sees what to do after a small mistake, so in
closed loop it drifts into states it has never seen. DART (Laskey et al., 2017) fixes this
by perturbing the expert while it records: the robot really moves a little off course, the
closed-loop expert corrects it, and the dataset keeps the expert's CLEAN command for every
(perturbed) state.

Isaac Lab's recorder saves the raw action given to env.step, before the action terms process
it. This class adds slowly varying (Ornstein-Uhlenbeck) noise to the PROCESSED commands right
after that, so the noise is executed but never recorded:
  - base: world-frame x/y speed and turn rate (the expert's base controller corrects drift)
  - right arm joints except the gripper (the expert re-aims from measured positions)
Left arm, gripper and head are left alone.

    noise = ActionNoise(env)                       # installs itself
    noise.base_enabled = expert.driving            # base noise only while the base is steered to a goal
    noise.arm_enabled = not expert.precise         # no arm noise while grasping or setting down
"""

import math

import torch

from xle_leisaac.robots.xlerobot import LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS


class ActionNoise:
    def __init__(self, env, base_std=(0.03, 0.03, 0.06), arm_std=0.015, tau=0.8):
        """base_std: m/s, m/s, rad/s; arm_std: rad; tau: correlation time (s)."""
        self.env = env
        self.base_enabled = self.arm_enabled = True
        manager = env.action_manager
        self.joints, self.base = manager._terms["joints"], manager._terms["base"]
        right = slice(len(LEFT_ARM_JOINTS), len(LEFT_ARM_JOINTS) + len(RIGHT_ARM_JOINTS) - 1)  # no gripper
        self.arm_slice = right
        dev = env.device
        self.base_std = torch.tensor(base_std, device=dev)
        self.arm_std = arm_std
        self.decay = math.exp(-env.step_dt / tau)
        self.gain = math.sqrt(1.0 - self.decay**2)  # keeps the stationary std at *_std
        self.base_noise = torch.zeros(3, device=dev)
        self.arm_noise = torch.zeros(right.stop - right.start, device=dev)
        original = manager.process_action

        def process_action(action):
            original(action)
            self._step()

        manager.process_action = process_action

    def reset(self):
        self.base_noise.zero_()
        self.arm_noise.zero_()

    def _step(self):
        self.base_noise = self.decay * self.base_noise + self.gain * self.base_std * torch.randn_like(self.base_noise)
        self.arm_noise = self.decay * self.arm_noise + self.gain * self.arm_std * torch.randn_like(self.arm_noise)
        if self.base_enabled:
            self.base.processed_actions[0] += self.base_noise
        if self.arm_enabled:
            self.joints.processed_actions[0, self.arm_slice] += self.arm_noise
