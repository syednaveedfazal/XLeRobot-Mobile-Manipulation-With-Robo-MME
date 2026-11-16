"""SO-101 leader arm -> sim follower joint angles, through LeRobot calibration files.

This is the same chain as real LeRobot teleoperation, with the follower simulated:

  leader servo ticks --(leader calibration)--> normalised value (what a real follower
  would be sent) --(follower calibration, xle_leisaac/robots/calibration.py)--> sim joint angle

- Normalised values use LeRobot's convention: arm joints -100..100 over the calibrated
  range, gripper 0..100. Values outside the range are clamped, and the leader's
  drive_mode = 1 flips the direction, as in lerobot/motors/motors_bus.py.
- Value -> sim angle is calibration.real_to_sim, the same conversion that turns recorded
  sim data into real-robot units (sim_to_lerobot_action) and that the Joy-Con /action
  input uses. So a leader pose, a Joy-Con command and a dataset value of the same number
  all put the sim arm in the same place. Arm joints: 0 = middle of the follower's
  calibrated range (LeRobot's degree mode). Gripper: 0..100 = closed..open.

Plain Python + numpy: no Isaac Sim needed, so tools/check_leader_mapping.py can test it.
"""

import importlib.util
import json
import os

import numpy as np

# Load calibration.py by path: importing the xle_leisaac package would need gymnasium,
# which the login node's plain Python (tools/check_leader_mapping.py) does not have.
_spec = importlib.util.spec_from_file_location(
    "xle_calibration", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "robots", "calibration.py")
)
calibration = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(calibration)

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
TICKS_PER_TURN = calibration.TICKS_PER_TURN  # 4095, as in LeRobot's degree mode


def load_calibration(path):
    with open(path) as f:
        cal = json.load(f)
    missing = [j for j in JOINTS if j not in cal]
    if missing:
        raise ValueError(f"{path}: missing joints {missing}")
    for j in JOINTS:
        if cal[j]["range_max"] <= cal[j]["range_min"]:
            raise ValueError(f"{path}: {j} has range_max <= range_min")
    return cal


def range_mid(motor):
    """Middle of a motor's calibrated range in ticks: 0 in LeRobot's normalised and degree units."""
    return (motor["range_min"] + motor["range_max"]) / 2


def normalize(joint, ticks, cal):
    """Leader ticks -> LeRobot normalised value (-100..100, gripper 0..100)."""
    c = cal[joint]
    lo, hi = c["range_min"], c["range_max"]
    frac = (min(hi, max(lo, ticks)) - lo) / (hi - lo)
    if joint == "gripper":
        value = frac * 100.0
        return 100.0 - value if c["drive_mode"] else value
    value = frac * 200.0 - 100.0
    return -value if c["drive_mode"] else value


class LeaderToSim:
    """Turns 6 leader servo readings into 6 sim joint targets (rad) for one follower arm."""

    def __init__(self, leader_cal, follower_cal):
        self.leader = leader_cal
        self.follower = follower_cal

    def values(self, ticks):
        """The 6 normalised values a real follower would be sent."""
        return np.array([normalize(joint, float(ticks[i]), self.leader) for i, joint in enumerate(JOINTS)])

    def __call__(self, ticks):
        # Values are clamped to the calibrated range, so the angles stay inside the joint limits.
        return calibration.real_to_sim(self.values(ticks)[None], self.follower)[0]
