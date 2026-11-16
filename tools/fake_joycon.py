"""fake_joycon.py - scripted stand-in for the `joyconrobotics` Joy-Con driver (tests only).

`import fake_joycon` registers a module named `joyconrobotics` whose JoyconRobotics class
behaves like the real driver's base class (a 100 Hz background loop calling update(), the
same attributes) but reads buttons and sticks from a time script instead of Bluetooth.
XLeRobot's FixedAxesJoyconRobotics subclass and the rest of its teleop code then run
unchanged on top of it. Used by `tools/joycon_sim_client.py --fake`.

Script (seconds since the Joy-Cons were created):
   3 -  6  right stick up        -> right arm reaches forward (IK)
   6 -  9  right X button        -> base moves (XLeRobot maps X to its 'backward' key)
   9 - 11  left D-pad left       -> head_motor_1 increases
  11 - 12  right ZR held         -> right gripper direction flips to "close"
  12.5-15  right ZR held again   -> direction flips to "open", gripper opens

It also aliases `lerobot.robots.so_follower` to the older `lerobot.robots.so101_follower`, so
XLeRobot's kinematics module imports with the LeRobot 0.4.2 inside the Isaac container. The
lab laptop's newer LeRobot does not need this.
"""

import sys
import threading
import time
import types

STICK_CENTRE = {"right": (1900, 2100), "left": (2300, 2000)}  # (vertical, horizontal), as XLeRobot's code expects
START = time.time()


def restart_clock():
    """Start the button script now (called when the control loop starts)."""
    global START
    START = time.time()


def _script(side, name):
    t = time.time() - START
    if side == "right" and name == "stick_v" and 3 <= t < 6:
        return STICK_CENTRE["right"][0] + 800
    if side == "right" and name == "x" and 6 <= t < 9:
        return 1
    if side == "left" and name == "left" and 9 <= t < 11:
        return 1
    if side == "right" and name == "zr" and (11 <= t < 12 or 12.5 <= t < 15):
        return 1
    if name == "stick_v":
        return STICK_CENTRE[side][0]
    if name == "stick_h":
        return STICK_CENTRE[side][1]
    return 0


class FakeJoyCon:
    def __init__(self, side):
        self.side = side

    def is_right(self):
        return self.side == "right"

    def is_left(self):
        return self.side == "left"

    def get_stick_right_vertical(self):
        return _script("right", "stick_v")

    def get_stick_right_horizontal(self):
        return _script("right", "stick_h")

    def get_stick_left_vertical(self):
        return _script("left", "stick_v")

    def get_stick_left_horizontal(self):
        return _script("left", "stick_h")

    def __getattr__(self, name):
        # get_button_x, get_button_zr, get_button_left, ... -> scripted 0/1 for this side
        if name.startswith("get_button_"):
            button = name[len("get_button_"):]
            return lambda: _script(self.side, button)
        raise AttributeError(name)

    def _close(self):
        pass


class FakeButtonEvents:
    def events(self):
        return []


class JoyconRobotics:
    """Same interface as joyconrobotics.JoyconRobotics, minus the hardware."""

    def __init__(self, device="right", gripper_state=1.0, dof_speed=(1, 1, 1, 1, 1, 1), offset_position_m=(0.0, 0.0, 0.0),
                 direction_reverse=(1, 1, 1), change_down_to_gripper=False, **_ignored):  # fmt: skip
        self.joycon = FakeJoyCon(device)
        self.button = FakeButtonEvents()
        self.gripper_state = gripper_state
        self.dof_speed = list(dof_speed)
        self.offset_position_m = list(offset_position_m)
        self.position = list(offset_position_m)
        self.orientation_rad = [0.0, 0.0, 0.0]
        self.posture = list(offset_position_m)
        self.direction_reverse = list(direction_reverse)
        self.change_down_to_gripper = change_down_to_gripper
        self.reset_button = self.next_episode_button = self.restart_episode_button = self.button_control = 0
        self.running = True
        self.lock = threading.Lock()
        threading.Thread(target=self.solve_loop, daemon=True).start()

    def get_orientation(self):
        return self.orientation_rad  # held flat on the table

    def reset_joycon(self):
        pass

    def update(self):  # as in the real driver
        roll, pitch, yaw = self.get_orientation()
        self.position, gripper, button_control = self.common_update()
        x, y, z = self.position
        self.posture = [x, y, z, roll, pitch, yaw]
        return self.posture, gripper, button_control

    def solve_loop(self):  # as in the real driver: 100 Hz, and an error just waits and retries
        while self.running:
            try:
                self.update()
                time.sleep(0.01)
            except Exception:  # e.g. the subclass has not finished __init__ yet
                time.sleep(0.05)

    def get_control(self, out_format="euler_rad"):
        roll, pitch, yaw = self.orientation_rad
        x, y, z = self.position
        self.posture = [x, y, z, roll, pitch, yaw]
        return self.posture, self.gripper_state, self.button_control

    def disconnect(self):
        self.running = False


sys.modules["joyconrobotics"] = types.SimpleNamespace(JoyconRobotics=JoyconRobotics)

# LeRobot 0.4.2 (Isaac container) names the SO-101 follower package so101_follower.
try:
    import lerobot.robots.so_follower.config_so_follower  # noqa: F401
except ImportError:
    import lerobot.robots.so101_follower.config_so101_follower as _old

    sys.modules["lerobot.robots.so_follower"] = types.ModuleType("lerobot.robots.so_follower")
    sys.modules["lerobot.robots.so_follower.config_so_follower"] = _old
