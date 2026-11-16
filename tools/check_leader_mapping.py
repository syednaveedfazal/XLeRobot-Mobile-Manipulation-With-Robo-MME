"""Print how the leader calibration maps to sim joint angles. No Isaac Sim needed.

    source env.local.sh && python3 tools/check_leader_mapping.py
    python3 tools/check_leader_mapping.py --leader calibration/my_leader.json --follower calibration/my_follower.json

For each sim arm (follower calibration from xle_leisaac/robots/calibration.py unless
--follower is given) and each joint: the leader's calibrated range (ticks) and the sim
angle at its min, at the calibration pose (ticks 2047) and at its max. Then it checks that
  - the leader's range ends land on the follower's joint limits,
  - with the same calibration on both sides, the sim copies the leader's angle (the
    calibration pose is sim angle 0),
  - leader -> sim angle -> real units (the dataset conversion) gives back the value the
    leader sent, so leader input and recorded data agree.
"""

import argparse
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "xle_leisaac", "teleop"))
from leader_arm import HOMED_TICKS, JOINTS, TICKS_PER_TURN, LeaderToSim, calibration, load_calibration, normalize  # noqa: E402


def check_arm(side, leader, follower, follower_name):
    to_sim = LeaderToSim(leader, follower)
    limits = calibration.joint_limits_deg(follower)
    same = leader == follower
    print(f"\n{side} arm, follower {follower_name}" + (" (same as leader)" if same else ""))
    print(f"{'joint':14s} {'leader ticks':>13s} {'sim at min':>10s} {'at calib':>9s} {'at max':>8s}   follower limits (deg)")
    ok = True
    for i, joint in enumerate(JOINTS):
        c = leader[joint]
        angles = []
        for ticks in (c["range_min"], HOMED_TICKS, c["range_max"]):
            sample = [HOMED_TICKS] * 6
            sample[i] = ticks
            angles.append(math.degrees(to_sim(sample)[i]))
        lo, hi = limits[joint]
        print(f"{joint:14s} {c['range_min']:5d}..{c['range_max']:<6d} {angles[0]:+9.1f} {angles[1]:+9.1f} {angles[2]:+8.1f}"
              f"   [{lo:+.1f}, {hi:+.1f}]")
        ok &= abs(angles[0] - lo) < 1e-6 and abs(angles[2] - hi) < 1e-6
        ok &= abs(normalize(joint, c["range_min"], leader) - (0 if joint == "gripper" else -100)) < 1e-9
        for ticks in np.linspace(c["range_min"], c["range_max"], 25):
            sample = [HOMED_TICKS] * 6
            sample[i] = ticks
            if same and joint != "gripper":  # same calibration: the sim copies the leader's angle
                ok &= abs(math.degrees(to_sim(sample)[i]) - (ticks - HOMED_TICKS) * 360 / TICKS_PER_TURN) < 1e-6
            # leader -> sim -> real units (dataset conversion) must return the value the leader sent
            back = calibration.sim_to_real(to_sim(sample)[None], follower)[0]
            ok &= bool(np.allclose(back, to_sim.values(sample), atol=1e-6))
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--leader", default="calibration/my_leader.json")
    p.add_argument("--follower", default=None, help="one follower file for both arms (default: calibration.py)")
    args = p.parse_args()

    leader = load_calibration(os.path.join(REPO, args.leader))
    print(f"leader {args.leader}")
    ok = True
    for side, path in calibration.ARM_CALIBRATION_FILES.items():
        path = os.path.join(REPO, args.follower) if args.follower else path
        follower = load_calibration(path)
        for problem in calibration.check(follower, os.path.relpath(path, REPO)):
            print("PROBLEM:", problem)
            ok = False
        ok &= check_arm(side, leader, follower, os.path.relpath(path, REPO))
    if leader["gripper"]["range_max"] - leader["gripper"]["range_min"] < 100:
        print("WARNING: gripper range is under 100 ticks; was the trigger moved during calibration?")
        ok = False
    print(f"\n[check] leader mapping {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
