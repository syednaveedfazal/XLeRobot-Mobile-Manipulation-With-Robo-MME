"""Turn the browser teleop's TeleopCommand into XLeRobot's 17-D Isaac action.

The browser page, keyboard and gamepad mappings come from the ManiSkill teleop
(xlerobot_sim/teleop_controller.py) and are reused unchanged. Its TeleopCommand says
*what the operator wants* (rates in [-1, 1]: base forward/turn, arm end-effector
up/down and forward/back, wrist pitch/roll, shoulder rotation, gripper, head).

This module turns that into our action, which matches the real robot:
  [0:6]  left arm joint targets (rad)     [6:12] right arm joint targets (rad)
  [12:14] head pan, tilt (rad)            [14:17] base forward m/s, left m/s, turn rad/s

Arm end-effector motion uses planar inverse kinematics on the official SO-101 URDF
(the ManiSkill controller's IK constants are for the SO-100 arm). The rate per step
is the same as in the ManiSkill controller, so both sims feel the same to drive.

Two things the real base can do that TeleopCommand has no field for are added here:
sideways driving (keyboard Q/E, gamepad View + D-pad left/right) and the real robot's
three speed levels (keys 1/2/3).
"""

import math
import os
import sys
import types

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# xlerobot_sim/__init__.py imports ManiSkill, which the Isaac container does not have.
# We only need two plain-Python modules from that package, so register the package
# without running its __init__ and import those modules directly.
if "xlerobot_sim" not in sys.modules:
    _pkg = types.ModuleType("xlerobot_sim")
    _pkg.__path__ = [os.path.join(REPO, "xlerobot_sim")]
    sys.modules["xlerobot_sim"] = _pkg
from xlerobot_sim.teleop_controller import TeleopCommand, TeleopController  # noqa: E402

# xlerobot_sim/teleop_web.py (whose page and HTTP server we reuse) imports `tyro` only for
# its own command line, which the Isaac version never calls. tyro needs a newer
# typing_extensions than the one Isaac Sim bundles, so give it an empty placeholder instead.
if "tyro" not in sys.modules:
    sys.modules["tyro"] = types.ModuleType("tyro")

sys.path.insert(0, os.path.join(REPO, "tools"))
from urdf_geometry import Urdf  # noqa: E402

SO101_URDF = os.path.join(os.environ["XLE_THIRD_PARTY"], "so_arm100_urdf/so101_new_calib_camera.urdf")
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")

# Real robot speed levels (XLeRobot software/src/robots/xlerobot/xlerobot.py, lines 57-59).
SPEED_LEVELS = ((0.1, 30.0), (0.2, 60.0), (0.3, 90.0))  # (m/s, deg/s)

# SO-101 new_calib gripper: -10 deg is closed, 100 deg fully open. Open to 80 deg.
GRIPPER_OPEN, GRIPPER_CLOSED = math.radians(80.0), math.radians(-10.0)
HEAD_LIMITS = ((-1.57, 1.57), (-0.76, 1.45))  # head pan, tilt (rad), from XLeRobot's URDF
STRAFE_KEYS = {"q": 1.0, "e": -1.0}  # +left
VIEW_BUTTON, DPAD_LEFT, DPAD_RIGHT = 8, 14, 15  # W3C standard gamepad indices


class So101PlanarArm:
    """Planar IK for one SO-101 arm, computed from the official URDF.

    In the arm's own vertical plane (shoulder pan at 0), shoulder lift and elbow place
    the wrist point (the wrist_flex axis) at (forward, up). Wrist flex then sets the
    gripper's pitch: the angle of the wrist-to-tool line in that plane.
    """

    def __init__(self):
        self.urdf = Urdf(SO101_URDF)
        self.limits = {}
        for name in JOINTS:
            limit = self.urdf.joints[name].find("limit")
            self.limits[name] = (float(limit.get("lower")), float(limit.get("upper")))
        # Wrist flex changes the gripper pitch one-for-one; find out in which direction.
        base = {"shoulder_lift": 0.0, "elbow_flex": 0.0}
        self.flex_sign = np.sign(self.pitch({**base, "wrist_flex": 0.1}) - self.pitch({**base, "wrist_flex": 0.0}))

    def wrist(self, q):
        p = self.urdf.joint_pose("wrist_flex", "base_link", q)[:3, 3]
        return np.array([p[0], p[2]])

    def pitch(self, q):
        wrist = self.urdf.joint_pose("wrist_flex", "base_link", q)[:3, 3]
        tool = self.urdf.link_pose("gripper_frame_link", "base_link", q)[:3, 3]
        return math.atan2(tool[2] - wrist[2], tool[0] - wrist[0])

    def solve(self, target, lift, elbow):
        """Shoulder lift and elbow that put the wrist at `target`, starting from the current values."""
        q = np.array([lift, elbow])
        lo = np.array([self.limits["shoulder_lift"][0], self.limits["elbow_flex"][0]])
        hi = np.array([self.limits["shoulder_lift"][1], self.limits["elbow_flex"][1]])

        def f(v):
            return self.wrist({"shoulder_lift": v[0], "elbow_flex": v[1]})

        for _ in range(8):  # damped least squares, a few steps are enough for small moves
            err = f(q) - target
            if np.linalg.norm(err) < 1e-4:
                break
            jac = np.column_stack([(f(q + d) - f(q)) / 1e-4 for d in (np.array([1e-4, 0]), np.array([0, 1e-4]))])
            q = np.clip(q - jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(2), err), lo, hi)
        return q

    def wrist_flex_for(self, pitch, lift, elbow):
        current = self.pitch({"shoulder_lift": lift, "elbow_flex": elbow, "wrist_flex": 0.0})
        lo, hi = self.limits["wrist_flex"]
        return float(np.clip((pitch - current) * self.flex_sign, lo, hi))


class IsaacTeleopAdapter:
    """Holds the joint targets and turns each TeleopCommand into a 17-D action."""

    def __init__(self, current_state, rest_arm):
        """current_state: 17-D state (sim units). rest_arm: 6 joint values the X / Menu reset goes to."""
        self.arm = So101PlanarArm()
        self.rest_arm = list(rest_arm)
        steps = TeleopController  # same per-step rates as the ManiSkill controller
        self.ee_step, self.rotation_step = steps.ee_step, steps.rotation_step
        self.pitch_step, self.roll_step, self.head_step = steps.pitch_step, steps.roll_step, steps.head_step
        self.speed = 1  # index into SPEED_LEVELS: start at medium
        self.gripper_closed = [False, False]  # [right, left] like the ManiSkill page; grippers start open
        self.reset(current_state)

    def reset(self, current_state):
        """Take the current robot state as the new targets (no jump)."""
        self.target = np.array(current_state[:14], dtype=float)
        self.ee, self.pitch = [], []
        for side in (0, 1):  # 0 = left arm (action 0:6), 1 = right arm (action 6:12)
            q = self._arm_q(side)
            self.ee.append(self.arm.wrist(q))
            self.pitch.append(self.arm.pitch(q))

    def _arm_q(self, side):
        o = 6 * side
        return dict(zip(JOINTS, self.target[o:o + 6]))

    def step(self, cmd: TeleopCommand, pressed=(), edges=(), gamepad=None):
        for key in ("1", "2", "3"):
            if key in edges:
                self.speed = int(key) - 1
        if cmd.reset_arms:
            self.target[0:6], self.target[6:12] = self.rest_arm, self.rest_arm
            self.reset(self.target)

        # TeleopCommand.arm1 is the RIGHT arm, arm2 the LEFT arm (ManiSkill page convention).
        for side, arm_cmd, closed_index in ((1, cmd.arm1, 0), (0, cmd.arm2, 1)):
            o = 6 * side
            self.ee[side] = self.ee[side] + self.ee_step * np.clip([arm_cmd.ee_x, arm_cmd.ee_y], -1, 1)
            self.pitch[side] += self.pitch_step * float(np.clip(arm_cmd.pitch, -1, 1))
            lift, elbow = self.arm.solve(self.ee[side], self.target[o + 1], self.target[o + 2])
            self.ee[side] = self.arm.wrist({"shoulder_lift": lift, "elbow_flex": elbow})  # no wind-up when out of reach
            self.target[o + 1], self.target[o + 2] = lift, elbow
            self.target[o + 3] = self.arm.wrist_flex_for(self.pitch[side], lift, elbow)
            for joint, rate, step in ((0, arm_cmd.rotation, self.rotation_step), (4, arm_cmd.roll, self.roll_step)):
                lo, hi = self.arm.limits[JOINTS[joint]]
                self.target[o + joint] = float(np.clip(self.target[o + joint] + step * np.clip(rate, -1, 1), lo, hi))
            if arm_cmd.toggle_gripper:
                self.gripper_closed[closed_index] = not self.gripper_closed[closed_index]
            closing = max(float(self.gripper_closed[closed_index]), float(np.clip(arm_cmd.squeeze, 0, 1)))
            self.target[o + 5] = GRIPPER_OPEN + closing * (GRIPPER_CLOSED - GRIPPER_OPEN)

        for i, rate in ((12, cmd.head_pan), (13, cmd.head_tilt)):
            lo, hi = HEAD_LIMITS[i - 12]
            self.target[i] = float(np.clip(self.target[i] + self.head_step * np.clip(rate, -1, 1), lo, hi))

        linear, angular = SPEED_LEVELS[self.speed]
        strafe = sum(v for k, v in STRAFE_KEYS.items() if k in pressed)
        if gamepad and gamepad.get("mapping") == "standard":
            buttons = list(gamepad["buttons"]) + [0.0] * 17
            if buttons[VIEW_BUTTON] > 0.5:  # View + D-pad left/right strafes instead of turning
                strafe += float(buttons[DPAD_LEFT] > 0.5) - float(buttons[DPAD_RIGHT] > 0.5)
                cmd.base_turn = 0.0
        base = [
            linear * float(np.clip(cmd.base_forward, -1, 1)),
            linear * float(np.clip(strafe, -1, 1)),
            math.radians(angular) * float(np.clip(cmd.base_turn, -1, 1)),
        ]
        return np.concatenate([self.target, base]).astype(np.float32)
