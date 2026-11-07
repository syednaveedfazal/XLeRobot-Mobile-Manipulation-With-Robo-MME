"""XLeRobot keyboard control, based on LeIsaac's lekiwi-keyboard.

Each arm key moves one joint while held, and Tab switches between the arms. LeIsaac's
SO-101 keyboard uses inverse kinematics on a single arm instead; joint moves were chosen
because our action is 17 joint-level values, so what you press is what gets recorded.

It needs a display or a livestream. Headless jobs cannot receive key presses.

  Arm (active arm):  A/D pan   W/S lift   Q/E elbow   K/I wrist pitch
                     J/L wrist roll   U/O gripper   Tab switch arm
  Base:              arrows drive   Z/X rotate   1/2/3 speed
  Head:              F/G pan   C/V tilt
  Session:           B start   R reset (fail)   N reset (success)
"""

import carb
import numpy as np
from leisaac.devices.device_base import Device

from xle_leisaac.robots.xlerobot import XLEROBOT_JOINT_NAMES

JOINT_STEP = 0.01  # radians per control step while a key is held

# key -> (index within one arm's 6 joints, direction)
ARM_KEYS = {
    "A": (0, 1), "D": (0, -1), "W": (1, -1), "S": (1, 1), "Q": (2, -1), "E": (2, 1),
    "K": (3, 1), "I": (3, -1), "J": (4, 1), "L": (4, -1), "U": (5, 1), "O": (5, -1),
}  # fmt: skip
# key -> (index in the 17-D action, direction)
HEAD_KEYS = {"F": (12, 1), "G": (12, -1), "C": (13, 1), "V": (13, -1)}
# key -> (forward, left, turn) direction
BASE_KEYS = {
    "UP": (1, 0, 0), "DOWN": (-1, 0, 0), "LEFT": (0, 1, 0), "RIGHT": (0, -1, 0), "Z": (0, 0, 1), "X": (0, 0, -1),
}  # fmt: skip
# Speed levels: (m/s, rad/s)
SPEEDS = ((0.1, np.radians(30)), (0.2, np.radians(60)), (0.3, np.radians(90)))


class XLeRobotKeyboard(Device):
    def __init__(self, env):
        super().__init__(env, "xlerobot-keyboard")
        self._held = set()
        self._active_arm = 1  # 0 is left, 1 is right
        self._speed = 0
        self._robot = env.scene["robot"]
        self._joint_ids = self._robot.find_joints(XLEROBOT_JOINT_NAMES, preserve_order=True)[0]
        self._target = None

    def _add_device_control_description(self):
        for key, text in (
            ("A/D", "arm pan"), ("W/S", "arm lift"), ("Q/E", "arm elbow"), ("K/I", "wrist pitch"),
            ("J/L", "wrist roll"), ("U/O", "gripper"), ("TAB", "switch arm"), ("ARROWS", "drive"),
            ("Z/X", "rotate"), ("1/2/3", "speed"), ("F/G", "head pan"), ("C/V", "head tilt"),
        ):  # fmt: skip
            self._display_controls_table.add_row([key, text])

    def get_device_state(self):
        """Return the 17-D action: 14 absolute joint targets and 3 body-frame base speeds."""
        if self._target is None:  # start from wherever the robot is
            self._target = self._robot.data.joint_pos[0, self._joint_ids[:14]].cpu().numpy().copy()

        for key in self._held:
            if key in ARM_KEYS:
                joint, direction = ARM_KEYS[key]
                self._target[self._active_arm * 6 + joint] += direction * JOINT_STEP
            elif key in HEAD_KEYS:
                index, direction = HEAD_KEYS[key]
                self._target[index] += direction * JOINT_STEP

        linear, angular = SPEEDS[self._speed]
        base = np.zeros(3)
        for key in self._held:
            if key in BASE_KEYS:
                base += np.array(BASE_KEYS[key]) * (linear, linear, angular)
        return np.concatenate([self._target, base])

    def reset(self):
        self._held.clear()
        self._target = None
        self._speed = 0

    def _on_keyboard_event(self, event, *args, **kwargs):
        super()._on_keyboard_event(event, *args, **kwargs)
        name = event.input.name
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if name == "TAB":
                self._active_arm = 1 - self._active_arm
                print(f"Active arm: {'left' if self._active_arm == 0 else 'right'}")
            elif name in ("KEY_1", "KEY_2", "KEY_3"):
                self._speed = int(name[-1]) - 1
            else:
                self._held.add(name)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._held.discard(name)
