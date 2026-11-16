"""Print how the leader calibration maps to sim joint angles. No Isaac Sim needed.

    source env.local.sh && python3 tools/check_leader_mapping.py
    python3 tools/check_leader_mapping.py --leader calibration/my_leader.json --follower calibration/my_follower.json

For each sim arm (follower calibration from xle_leisaac/robots/calibration.py unless
--follower is given) and each joint: the leader's calibrated range (ticks) and the sim
angle at its min, middle and max. Then it checks that
  - the leader's range ends land on the follower's joint limits and its middle on 0,
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
from leader_arm import JOINTS, LeaderToSim, calibration, load_calibration, normalize, range_mid  # noqa: E402


def check_arm(side, leader, follower, follower_name):
    to_sim = LeaderToSim(leader, follower)
    limits = calibration.joint_limits_deg(follower)
    mids = [range_mid(leader[j]) for j in JOINTS]
    print(f"\n{side} arm, follower {follower_name}")
    print(f"{'joint':14s} {'leader ticks':>13s} {'sim at min':>10s} {'at mid':>8s} {'at max':>8s}   follower limits (deg)")
    ok = True
    for i, joint in enumerate(JOINTS):
        c = leader[joint]
        angles = []
        for ticks in (c["range_min"], mids[i], c["range_max"]):
            sample = list(mids)
            sample[i] = ticks
            angles.append(math.degrees(to_sim(sample)[i]))
        lo, hi = limits[joint]
        print(f"{joint:14s} {c['range_min']:5d}..{c['range_max']:<6d} {angles[0]:+9.1f} {angles[1]:+8.1f} {angles[2]:+8.1f}"
              f"   [{lo:+.1f}, {hi:+.1f}]")
        ends = (hi, lo) if c["drive_mode"] else (lo, hi)
        ok &= abs(angles[0] - ends[0]) < 1e-6 and abs(angles[2] - ends[1]) < 1e-6
        if joint != "gripper":
            ok &= abs(angles[1]) < 1e-6
        ok &= abs(normalize(joint, c["range_min"], leader) - (0 if joint == "gripper" else -100)) < 1e-9 or c["drive_mode"]
        # leader -> sim -> real units (dataset conversion) must return the value the leader sent
        for ticks in np.linspace(c["range_min"], c["range_max"], 25):
            sample = list(mids)
            sample[i] = ticks
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
        ok &= check_arm(side, leader, load_calibration(path), os.path.relpath(path, REPO))
    if leader["gripper"]["range_max"] - leader["gripper"]["range_min"] < 100:
        print("WARNING: gripper range is under 100 ticks; was the trigger moved during calibration?")
        ok = False
    print(f"\n[check] leader mapping {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
