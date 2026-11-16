"""Real-robot calibration (LeRobot JSON) -> exact sim joint ranges and unit conversion.

A LeRobot calibration file stores, per motor, the encoder range the joint really moves
through (range_min .. range_max, 12-bit ticks, 4096 per turn). Two facts from LeRobot
(lerobot/motors/motors_bus.py) fix where every number puts the arm:

  - Calibration ("set_half_turn_homings") makes the pose you hold when asked for "the
    middle of its range of motion" read HOMED_TICKS = 2047 on every joint.
  - Recorded values (RANGE_M100_100, the SO-101 default) are linear over the calibrated
    range:  value = (ticks - min) / (max - min) * 200 - 100  (gripper: 0..100),
    clamped to that range. So value 0 is the middle of min..max, which is in general NOT
    the calibration pose (on our follower, shoulder lift and elbow differ by ~20 deg).

The official SO-101 "new_calib" URDF that the sim uses has its joint zero at that same
calibration pose. So the exact conversion goes through the servo ticks:

    value -> ticks = min + (value + 100) / 200 * (max - min)
          -> sim angle (deg) = (ticks - 2047) * 360 / 4096

and the joint's real reachable range is (min - 2047) .. (max - 2047) ticks from the
calibration pose, in general not symmetric. The gripper keeps LeIsaac's convention
(0 = closed = -10 deg, 100 = fully open = 100 deg), because its leader has a trigger,
not jaws.

Plain Python + json + numpy, so build scripts can use it without Isaac Sim.
"""

import json
import os

import numpy as np

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
TICKS_PER_TURN = 4096  # STS3215 encoder resolution
HOMED_TICKS = 2047  # what LeRobot's half-turn homing makes the calibration pose read
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


def _ticks_to_deg(ticks):
    return (ticks - HOMED_TICKS) * 360.0 / TICKS_PER_TURN


def joint_limits_deg(calibration):
    """{joint: (lower, upper)} in degrees from the calibration pose, the arm's real reachable range."""
    limits = {name: (_ticks_to_deg(calibration[name]["range_min"]), _ticks_to_deg(calibration[name]["range_max"]))
              for name in ARM_JOINTS}  # fmt: skip
    limits["gripper"] = GRIPPER_DEG
    return limits


def real_to_sim(values, calibration):
    """(N, 6) LeRobot values for one arm -> (N, 6) sim joint angles in radians.
    Values are clamped to -100..100 (gripper 0..100), as LeRobot does before moving a motor."""
    values = np.asarray(values, dtype=np.float64)
    deg = np.zeros_like(values)
    for i, name in enumerate(ARM_JOINTS):
        lo, hi = calibration[name]["range_min"], calibration[name]["range_max"]
        ticks = lo + (np.clip(values[:, i], -100.0, 100.0) + 100.0) / 200.0 * (hi - lo)
        deg[:, i] = _ticks_to_deg(ticks)
    low, high = GRIPPER_DEG
    deg[:, 5] = low + np.clip(values[:, 5], 0.0, 100.0) / 100.0 * (high - low)
    return np.radians(deg)


def sim_to_real(radians, calibration):
    """(N, 6) sim joint angles in radians for one arm -> (N, 6) LeRobot values.
    Angles beyond the calibrated range read as its end, as LeRobot does when reading a motor."""
    deg = np.degrees(np.asarray(radians, dtype=np.float64))
    values = np.zeros_like(deg)
    for i, name in enumerate(ARM_JOINTS):
        lo, hi = calibration[name]["range_min"], calibration[name]["range_max"]
        ticks = np.clip(HOMED_TICKS + deg[:, i] * TICKS_PER_TURN / 360.0, lo, hi)
        values[:, i] = (ticks - lo) / (hi - lo) * 200.0 - 100.0
    low, high = GRIPPER_DEG
    values[:, 5] = np.clip((deg[:, 5] - low) / (high - low) * 100.0, 0.0, 100.0)
    return values


def describe(calibration):
    """Text table of the ranges, for logs and docs."""
    rows = []
    limits = joint_limits_deg(calibration)
    zero = np.degrees(real_to_sim(np.zeros((1, 6)), calibration)[0])
    for i, name in enumerate(ARM_JOINTS):
        m = calibration[name]
        lo, hi = limits[name]
        rows.append(f"{name:14} ticks {m['range_min']:5}..{m['range_max']:5} -> {lo:+7.1f}..{hi:+7.1f} deg "
                    f"from the calibration pose (value 0 = {zero[i]:+6.1f} deg)")
    m = calibration["gripper"]
    rows.append(f"{'gripper':14} ticks {m['range_min']:5}..{m['range_max']:5} -> value 0..100 = {GRIPPER_DEG[0]}..{GRIPPER_DEG[1]} deg")
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
        if joint != "gripper" and not m["range_min"] <= HOMED_TICKS <= m["range_max"]:
            problems.append(f"{name}: {joint} calibration pose ({HOMED_TICKS}) is outside its range")
        if m.get("drive_mode", 0):
            problems.append(f"{name}: {joint} has drive_mode 1, which the sim conversion does not handle")
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
        home = sim_to_real(np.zeros((1, 6)), cal)
        print("calibration pose (sim all zeros) back to sim:", bool(np.allclose(real_to_sim(home, cal)[0, :5], 0.0)))
