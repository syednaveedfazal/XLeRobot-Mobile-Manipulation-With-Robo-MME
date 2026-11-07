"""Teleop -> action mapping for the dual-arm XLeRobot in ManiSkill.

Input devices (keyboard, gamepad, later Joy-Con) are first turned into a device-independent
`TeleopCommand` (rates in [-1, 1] per degree of freedom). `TeleopController` then turns
commands into actions with the control law of upstream
`third_party/xlerobot_maniskill/examples/demo_ctrl_action_ee_keyboard.py`
(planar IK for Pitch/Elbow, P-control on joint targets).

Control mode: ``pd_joint_delta_pos_dual_arm`` (16-dim action):
    [0] base forward velocity    [1] base rotation
    [2:7]  arm 1 = RIGHT arm (Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll)
    [7:12] arm 2 = LEFT arm  (same joints, suffix _2)
    [12] gripper 1  [13] gripper 2  [14] head pan  [15] head tilt
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np

ACTION_DIM = 16
GRIPPER_OPEN, GRIPPER_CLOSED = 2.5, 0.1  # Jaw joint targets used upstream


# --------------------------------------------------------------------------- #
# Kinematics helpers (from upstream)
# --------------------------------------------------------------------------- #
def inverse_kinematics(x: float, y: float, l1: float = 0.1159, l2: float = 0.1350):
    """2-link planar IK for SO-101 (Pitch, Elbow) with URDF joint offsets."""
    theta1_offset = math.atan2(0.028, 0.11257)
    theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset

    r = math.sqrt(x**2 + y**2)
    r_max = l1 + l2
    if r > r_max:
        x, y, r = x * r_max / r, y * r_max / r, r_max
    r_min = abs(l1 - l2)
    if 0 < r < r_min:
        x, y, r = x * r_min / r, y * r_min / r, r_min

    cos_theta2 = -(r**2 - l1**2 - l2**2) / (2 * l1 * l2)
    theta2 = math.pi - math.acos(max(-1.0, min(1.0, cos_theta2)))
    beta = math.atan2(y, x)
    gamma = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
    theta1 = beta + gamma

    joint2 = max(-0.1, min(3.45, theta1 + theta1_offset))
    joint3 = max(-0.2, min(math.pi, theta2 + theta2_offset))
    return joint2, joint3


def get_mapped_joints(robot) -> np.ndarray:
    """Map the robot's 17 qpos entries to the 16-dim action layout."""
    if robot is None:
        return np.zeros(ACTION_DIM)
    q = robot.get_qpos()
    if hasattr(q, "cpu"):
        q = q.cpu()
    q = np.asarray(q).squeeze()
    m = np.zeros(ACTION_DIM)
    if len(q) >= 17:
        m[0], m[1] = q[0], q[2]
        m[2:7] = q[[3, 6, 9, 11, 13]]
        m[7:12] = q[[4, 7, 10, 12, 14]]
        m[12], m[13] = q[15], q[16]
        m[14], m[15] = q[5], q[8]
    return m


# --------------------------------------------------------------------------- #
# Device-independent command
# --------------------------------------------------------------------------- #
@dataclass
class ArmCommand:
    ee_x: float = 0.0      # end effector forward(+)/back(-)
    ee_y: float = 0.0      # end effector up(+)/down(-)
    rotation: float = 0.0  # shoulder pan
    pitch: float = 0.0     # wrist pitch
    roll: float = 0.0      # wrist roll
    squeeze: float = 0.0   # analog gripper closing in [0, 1] (0 = use toggle state)
    toggle_gripper: bool = False


@dataclass
class TeleopCommand:
    base_forward: float = 0.0
    base_turn: float = 0.0  # +left
    head_pan: float = 0.0
    head_tilt: float = 0.0
    arm1: ArmCommand = field(default_factory=ArmCommand)  # right arm
    arm2: ArmCommand = field(default_factory=ArmCommand)  # left arm
    reset_arms: bool = False

    def __add__(self, o: "TeleopCommand") -> "TeleopCommand":
        def arm(a: ArmCommand, b: ArmCommand) -> ArmCommand:
            return ArmCommand(a.ee_x + b.ee_x, a.ee_y + b.ee_y, a.rotation + b.rotation, a.pitch + b.pitch,
                              a.roll + b.roll, max(a.squeeze, b.squeeze), a.toggle_gripper or b.toggle_gripper)
        return TeleopCommand(self.base_forward + o.base_forward, self.base_turn + o.base_turn,
                             self.head_pan + o.head_pan, self.head_tilt + o.head_tilt,
                             arm(self.arm1, o.arm1), arm(self.arm2, o.arm2), self.reset_arms or o.reset_arms)


# --------------------------------------------------------------------------- #
# Controller (command -> action)
# --------------------------------------------------------------------------- #
class TeleopController:
    """Integrates commands into joint targets and P-controls toward them."""

    tip_length = 0.108
    # per-step increments at full deflection (same as upstream keyboard demo)
    ee_step = 0.005
    rotation_step = 0.01
    pitch_step = 0.02
    roll_step = 0.03
    turn_step = 0.02
    head_step = 0.02
    base_speed = 0.1
    warmup_steps = 50
    initial_ee = (0.162, 0.118)

    def __init__(self):
        self.p_gain = np.ones(ACTION_DIM)
        self.p_gain[0] = 2.0
        self.p_gain[1] = 0.5
        self.p_gain[12:14] = 0.05
        self.p_gain[14:16] = 2.0
        self.step_counter = 0
        self.gripper_closed = [False, False]  # toggle state per arm; grippers start open
        self.reset()

    def reset(self, current_joints: Optional[np.ndarray] = None):
        """Reset arm targets to the upstream initial pose. Keeps base heading and head."""
        self.ee = [np.array(self.initial_ee, dtype=float), np.array(self.initial_ee, dtype=float)]
        self.pitch = [0.0, 0.0]
        t = np.zeros(ACTION_DIM)
        if current_joints is not None:
            t[1] = current_joints[1]
            t[14:16] = current_joints[14:16]
        t[6] = t[11] = 1.57
        self.target = t
        self._apply_arm_ik(0)
        self._apply_arm_ik(1)

    def _apply_arm_ik(self, i: int):
        o = 2 + 5 * i  # first joint index of this arm
        x, y = self.ee[i]
        self.target[o + 1], self.target[o + 2] = inverse_kinematics(x, y + self.tip_length * math.sin(self.pitch[i]))
        self.target[o + 3] = self.target[o + 1] - self.target[o + 2] + self.pitch[i]

    @property
    def warming_up(self) -> bool:
        return self.step_counter < self.warmup_steps

    def step(self, cmd: TeleopCommand, current_joints: np.ndarray) -> np.ndarray:
        action = np.zeros(ACTION_DIM)
        self.step_counter += 1
        if self.step_counter <= self.warmup_steps:
            return action
        if cmd.reset_arms:
            self.reset(current_joints)
        t = self.target

        action[0] = float(np.clip(cmd.base_forward, -1, 1)) * self.base_speed
        t[1] += float(np.clip(cmd.base_turn, -1, 1)) * self.turn_step
        t[14] += float(np.clip(cmd.head_pan, -1, 1)) * self.head_step
        t[15] += float(np.clip(cmd.head_tilt, -1, 1)) * self.head_step

        for i, a in enumerate((cmd.arm1, cmd.arm2)):
            o = 2 + 5 * i
            self.ee[i][0] += float(np.clip(a.ee_x, -1, 1)) * self.ee_step
            self.ee[i][1] += float(np.clip(a.ee_y, -1, 1)) * self.ee_step
            self.pitch[i] += float(np.clip(a.pitch, -1, 1)) * self.pitch_step
            t[o] += float(np.clip(a.rotation, -1, 1)) * self.rotation_step
            t[o + 4] += float(np.clip(a.roll, -1, 1)) * self.roll_step
            self._apply_arm_ik(i)

            if a.toggle_gripper:
                self.gripper_closed[i] = not self.gripper_closed[i]
            closed = max(1.0 if self.gripper_closed[i] else 0.0, float(np.clip(a.squeeze, 0, 1)))
            t[12 + i] = GRIPPER_OPEN + closed * (GRIPPER_CLOSED - GRIPPER_OPEN)

        action[1:] = self.p_gain[1:] * (t[1:] - current_joints[1:])
        return action


# --------------------------------------------------------------------------- #
# Keyboard (keys are browser KeyboardEvent.key values, lowercased)
# --------------------------------------------------------------------------- #
KEY_HELP = [
    ("Base", [("W / S", "forward / back"), ("A / D", "turn left / right")]),
    ("Arm 1 (right)", [("8 / U", "EE up / down"), ("9 / I", "EE forward / back"),
                       ("7 / Y", "shoulder rotation +/-"), ("0 / O", "wrist pitch +/-"),
                       ("- / P", "wrist roll +/-"), ("V", "toggle gripper")]),
    ("Arm 2 (left)", [("J / M", "EE up / down"), ("K / ,", "EE forward / back"),
                      ("H / N", "shoulder rotation +/-"), ("L / .", "wrist pitch +/-"),
                      ("; / /", "wrist roll +/-"), ("B", "toggle gripper")]),
    ("Head", [("R / T", "pan +/-"), ("F / G", "tilt +/-")]),
    ("Other", [("X", "reset arm targets"), ("Esc", "release all keys")]),
]


def keyboard_command(pressed: Iterable[str], newly_pressed: Iterable[str]) -> TeleopCommand:
    k = {s.lower() for s in pressed}
    e = {s.lower() for s in newly_pressed}

    def ax(pos: str, neg: str) -> float:
        return float(pos in k) - float(neg in k)

    return TeleopCommand(
        base_forward=ax("w", "s"),
        base_turn=ax("a", "d"),
        head_pan=ax("r", "t"),
        head_tilt=ax("f", "g"),
        arm1=ArmCommand(ee_x=ax("9", "i"), ee_y=ax("8", "u"), rotation=ax("7", "y"),
                        pitch=ax("0", "o"), roll=ax("-", "p"), toggle_gripper="v" in e),
        arm2=ArmCommand(ee_x=ax("k", ","), ee_y=ax("j", "m"), rotation=ax("h", "n"),
                        pitch=ax("l", "."), roll=ax(";", "/"), toggle_gripper="b" in e),
        reset_arms="x" in e,
    )


# --------------------------------------------------------------------------- #
# Gamepad: W3C "standard" mapping (Xbox, PlayStation, Switch Pro, ... in the browser)
# --------------------------------------------------------------------------- #
GAMEPAD_HELP = [
    ("Base", [("D-pad up / down", "forward / back"), ("D-pad left / right", "turn")]),
    ("Arm 1 (right)", [("Right stick", "EE up-down / forward-back"), ("RB + right stick", "wrist pitch / roll"),
                       ("R3 (click) + right stick", "shoulder rotation"), ("RT", "close gripper (analog)")]),
    ("Arm 2 (left)", [("Left stick", "EE up-down / forward-back"), ("LB + left stick", "wrist pitch / roll"),
                      ("L3 (click) + left stick", "shoulder rotation"), ("LT", "close gripper (analog)")]),
    ("Head", [("X / B", "pan left / right"), ("Y / A", "tilt up / down")]),
    ("Other", [("Menu (≡)", "reset arm targets")]),
]

# standard-mapping indices
A, B, X, Y, LB, RB, LT, RT, VIEW, MENU, L3, R3, UP, DOWN, LEFT, RIGHT = range(16)
LX, LY, RX, RY = range(4)


def _dz(v: float, deadzone: float) -> float:
    """Deadzone with rescaling so output still reaches +-1."""
    if abs(v) < deadzone:
        return 0.0
    return math.copysign((abs(v) - deadzone) / (1 - deadzone), v)


def gamepad_command(axes: Sequence[float], buttons: Sequence[float], prev_buttons: Sequence[float],
                    deadzone: float = 0.15) -> TeleopCommand:
    """Map a standard-layout gamepad (browser Gamepad API) to a command.

    axes: [LX, LY, RX, RY] in [-1, 1], stick up = -1.  buttons: values in [0, 1].
    """
    axes = list(axes) + [0.0] * (4 - len(axes))
    buttons = list(buttons) + [0.0] * (17 - len(buttons))
    prev = list(prev_buttons) + [0.0] * (17 - len(prev_buttons))

    def held(i):
        return buttons[i] > 0.5

    def arm(stick_x, stick_y, bumper, click, trigger) -> ArmCommand:
        sx, sy = _dz(axes[stick_x], deadzone), -_dz(axes[stick_y], deadzone)  # sy: up = +
        c = ArmCommand(squeeze=buttons[trigger])
        if held(click):
            c.rotation = sx  # stick right -> arm swings to the robot's right (verified in smoke_test)
        elif held(bumper):
            c.pitch, c.roll = sy, sx
        else:
            c.ee_x, c.ee_y = sx, sy
        return c

    return TeleopCommand(
        base_forward=float(held(UP)) - float(held(DOWN)),
        base_turn=float(held(LEFT)) - float(held(RIGHT)),
        head_pan=float(held(X)) - float(held(B)),
        head_tilt=float(held(Y)) - float(held(A)),
        arm1=arm(RX, RY, RB, R3, RT),
        arm2=arm(LX, LY, LB, L3, LT),
        reset_arms=held(MENU) and prev[MENU] <= 0.5,
    )
