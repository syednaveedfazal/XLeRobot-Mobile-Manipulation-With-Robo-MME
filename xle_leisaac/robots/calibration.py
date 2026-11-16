"""Real-robot calibration (LeRobot JSON) -> exact sim joint ranges and unit conversion.

A LeRobot calibration file stores, per motor, the encoder range the joint really moves
through (range_min .. range_max, 12-bit ticks, 4096 per turn). LeRobot turns encoder
ticks into the numbers in recorded data like this (lerobot/motors/motors_bus.py, _normalize):

    arm joints (RANGE_M100_100):  value = (raw - min) / (max - min) * 200 - 100
    gripper    (RANGE_0_100):     value = (raw - min) / (max - min) * 100
    degree mode (DEGREES):        value = (raw - mid) * 360 / 4095,   mid = (min + max) / 2

So 0 always means "middle of this arm's calibrated range". The SO-101 "new_calib" URDF is
meant to be used with LeRobot's degree mode, so its joint angle 0 is that middle too.
For the 5 arm joints that gives an exact, linear conversion with a per-joint scale:

    sim angle (deg) = value / 100 * half_range_deg,   half_range_deg = (max - min) / 2 * 360 / 4095

and the joint's real reachable range is +-half_range_deg. The gripper keeps LeIsaac's
convention (0 = closed = -10 deg, 100 = fully open = 100 deg), because the URDF's gripper
zero is the closed jaw, not the middle of its range.

Plain Python + json + numpy, so build scripts can use it without Isaac Sim.
"""

import json
import os

import numpy as np

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
TICKS_PER_TURN = 4095  # LeRobot's max_res for the STS3215 (resolution 4096 - 1)
GRIPPER_DEG = (-10.0, 100.0)  # sim gripper angle at value 0 (closed) and 100 (open)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Which calibration file each sim arm uses. Only one follower arm has been calibrated so
# far, so both arms use it. Replace with per-arm files once the XLeRobot's own left and
# right arms are calibrated.
ARM_CALIBRATION_FILES = {
    "left": os.path.join(REPO, "calibration/my_follower.json"),
    "right": os.path.join(REPO, "calibration/my_follower.json"),
}


def load(path):
    with open(path) as f:
        return json.load(f)


def half_range_deg(motor):
    return (motor["range_max"] - motor["range_min"]) / 2 * 360 / TICKS_PER_TURN


def joint_limits_deg(calibration):
    """{joint: (lower, upper)} in degrees, the arm's real reachable range."""
    limits = {name: (-half_range_deg(calibration[name]), half_range_deg(calibration[name])) for name in ARM_JOINTS}
    limits["gripper"] = GRIPPER_DEG
    return limits


def real_to_sim(values, calibration):
    """(N, 6) LeRobot values for one arm -> (N, 6) sim joint angles in radians."""
    values = np.asarray(values, dtype=np.float64)
    deg = np.zeros_like(values)
    for i, name in enumerate(ARM_JOINTS):
        deg[:, i] = values[:, i] / 100.0 * half_range_deg(calibration[name])
    low, high = GRIPPER_DEG
    deg[:, 5] = low + values[:, 5] / 100.0 * (high - low)
    return np.radians(deg)


def sim_to_real(radians, calibration):
    """(N, 6) sim joint angles in radians for one arm -> (N, 6) LeRobot values."""
    deg = np.degrees(np.asarray(radians, dtype=np.float64))
    values = np.zeros_like(deg)
    for i, name in enumerate(ARM_JOINTS):
        values[:, i] = deg[:, i] / half_range_deg(calibration[name]) * 100.0
    low, high = GRIPPER_DEG
    values[:, 5] = (deg[:, 5] - low) / (high - low) * 100.0
    return values


def describe(calibration):
    """Text table of the ranges, for logs and docs."""
    rows = []
    for name in ARM_JOINTS:
        m = calibration[name]
        mid = (m["range_min"] + m["range_max"]) / 2
        rows.append(f"{name:14} ticks {m['range_min']:5}..{m['range_max']:5} (mid {mid:7.1f})  -> +-{half_range_deg(m):6.1f} deg")
    m = calibration["gripper"]
    rows.append(f"{'gripper':14} ticks {m['range_min']:5}..{m['range_max']:5}           -> value 0..100 = {GRIPPER_DEG[0]}..{GRIPPER_DEG[1]} deg")
    return "\n".join(rows)


def check(calibration, name="calibration"):
    """Problems that make a calibration unusable or suspicious (empty list = fine)."""
    problems = []
    for joint in ARM_JOINTS + ("gripper",):
        m = calibration.get(joint)
        if m is None:
            problems.append(f"{name}: no entry for {joint}")
            continue
        span = m["range_max"] - m["range_min"]
        if span < 200:  # ~18 degrees: the joint was hardly moved while calibrating
            problems.append(f"{name}: {joint} range is only {span} ticks ({span * 360 / TICKS_PER_TURN:.1f} deg)")
    return problems


if __name__ == "__main__":
    for side, path in ARM_CALIBRATION_FILES.items():
        cal = load(path)
        print(f"{side} arm: {os.path.relpath(path, REPO)}")
        print(describe(cal))
        print("problems:", check(cal, os.path.basename(path)) or "none")
        values = np.array([[100.0, -100.0, 50.0, -20.0, 10.0, 70.0]])
        back = sim_to_real(real_to_sim(values, cal), cal)
        print("round trip real -> sim -> real exact:", bool(np.allclose(back, values)))
