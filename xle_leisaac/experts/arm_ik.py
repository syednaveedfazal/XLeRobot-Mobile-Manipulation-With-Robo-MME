"""Top-down inverse kinematics for one XLeRobot arm, from the built robot URDF.

Given a target point for the gripper's tool point (right_arm_gripper_frame_link, between
the fingertips) in the robot's base_link frame, find shoulder pan, shoulder lift, elbow and
wrist flex so the tool point is there and the gripper points straight down. Wrist roll and
the gripper are left to the caller.

It is Gauss-Newton on the URDF's forward kinematics, the same file the sim's USD is built
from ($XLE_ASSETS/robots/xlerobot_so101/xlerobot_so101.urdf), so it matches the sim exactly,
including the calibrated joint limits. Plain numpy, no Isaac Sim needed.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools"))
from urdf_geometry import Urdf  # noqa: E402

URDF = os.path.join(os.environ["XLE_ASSETS"], "robots/xlerobot_so101/xlerobot_so101.urdf")
SOLVED_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")


class TopDownArmIK:
    def __init__(self, side="right", urdf_path=URDF):
        self.urdf = Urdf(urdf_path)
        self.side = side
        self.names = [f"{side}_arm_{j}" for j in SOLVED_JOINTS]
        limits = [self.urdf.joints[n].find("limit") for n in self.names]
        self.low = np.array([float(l.get("lower")) for l in limits])
        self.high = np.array([float(l.get("upper")) for l in limits])

    def tool_and_direction(self, q):
        """Tool point and pointing direction (wrist -> tool, unit) in base_link for joints q (4,)."""
        joints = dict(zip(self.names, q))
        tool = self.urdf.link_pose(f"{self.side}_arm_gripper_frame_link", "base_link", joints)[:3, 3]
        wrist = self.urdf.joint_pose(f"{self.side}_arm_wrist_flex", "base_link", joints)[:3, 3]
        direction = tool - wrist
        return tool, direction / np.linalg.norm(direction)

    def _residual(self, q, target):
        tool, direction = self.tool_and_direction(q)
        # position error, plus "point down" with a weight that makes 1 cm ~ 3 degrees
        return np.concatenate([tool - target, 0.05 * (direction - np.array([0.0, 0.0, -1.0]))])

    def solve(self, target, q0, iterations=30):
        """Joint values (4,) for `target` (base_link, metres), starting from q0 (4,).

        Returns (q, position_error_m). Joints stay inside the calibrated limits; if the target
        is out of reach the result is the closest reachable pose.
        """
        q = np.clip(np.asarray(q0, dtype=float), self.low, self.high)
        for _ in range(iterations):
            r = self._residual(q, target)
            if np.linalg.norm(r[:3]) < 5e-4 and np.linalg.norm(r[3:]) < 5e-4:
                break
            jac = np.empty((6, 4))
            for i in range(4):
                dq = np.zeros(4)
                dq[i] = 1e-4
                jac[:, i] = (self._residual(q + dq, target) - r) / 1e-4
            step = np.linalg.solve(jac.T @ jac + 1e-4 * np.eye(4), -jac.T @ r)  # damped least squares
            q = np.clip(q + step, self.low, self.high)
        tool, _ = self.tool_and_direction(q)
        return q, float(np.linalg.norm(tool - target))
