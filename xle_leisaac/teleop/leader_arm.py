"""SO-101 leader arm -> sim follower joint angles, through LeRobot calibration files.

This is the same chain as real LeRobot teleoperation, with the follower simulated:

  leader servo ticks --(leader calibration)--> normalised value --(follower calibration)-->
  follower servo ticks --> joint angle in the sim

- Normalised values use LeRobot's convention: arm joints -100..100 over the calibrated
  range, gripper 0..100. Values outside the range are clamped, and drive_mode = 1 flips
  the direction, as in lerobot/motors/motors_bus.py.
- Arm joint angle: LeRobot calibration ("set_half_turn_homings") makes the pose you hold
  when you press ENTER read 2047 ticks. The official SO-101 "new_calib" model, which the
  sim uses, is defined so that this same pose is all joints at 0. So
  angle = (ticks - 2047) * 360 / 4096 degrees.
- Gripper: the leader has a trigger, not jaws, so its ticks do not mean the same jaw
  angle. Its 0..100 range is spread over the sim gripper's range (closed .. open) instead.

With the follower calibration equal to the leader's (calibration/my_follower.json is a
copy of my_leader.json), the follower joints copy the leader joints one to one.

Plain Python + numpy: no Isaac Sim needed, so tools/check_leader_mapping.py can test it.
"""

import json
import math

import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
TICKS_PER_TURN = 4096
HOMED_TICKS = 2047  # what LeRobot's half-turn homing makes the calibration pose read


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


def unnormalize(joint, value, cal):
    """LeRobot normalised value -> follower ticks (float, not rounded)."""
    c = cal[joint]
    lo, hi = c["range_min"], c["range_max"]
    if joint == "gripper":
        value = 100.0 - value if c["drive_mode"] else value
        frac = min(100.0, max(0.0, value)) / 100.0
    else:
        value = -value if c["drive_mode"] else value
        frac = (min(100.0, max(-100.0, value)) + 100.0) / 200.0
    return lo + frac * (hi - lo)


class LeaderToSim:
    """Turns 6 leader servo readings into 6 sim joint targets (rad)."""

    def __init__(self, leader_cal, follower_cal, joint_limits):
        """joint_limits: joint name -> (lower, upper) in rad, from the sim's SO-101 model."""
        self.leader = leader_cal
        self.follower = follower_cal
        self.limits = joint_limits

    def __call__(self, ticks):
        q = np.zeros(len(JOINTS))
        for i, joint in enumerate(JOINTS):
            value = normalize(joint, float(ticks[i]), self.leader)
            follower_ticks = unnormalize(joint, value, self.follower)
            lo, hi = self.limits[joint]
            if joint == "gripper":
                c = self.follower[joint]
                frac = (follower_ticks - c["range_min"]) / (c["range_max"] - c["range_min"])
                q[i] = lo + frac * (hi - lo)  # 0 = closed (lower limit), 100 = open
            else:
                angle = math.radians((follower_ticks - HOMED_TICKS) * 360.0 / TICKS_PER_TURN)
                q[i] = min(hi, max(lo, angle))
        return q
