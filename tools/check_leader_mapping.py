"""Print how the leader calibration maps to sim joint angles. No Isaac Sim needed.

    source env.local.sh && python3 tools/check_leader_mapping.py
    python3 tools/check_leader_mapping.py --leader calibration/my_leader.json --follower calibration/my_follower.json

For each joint: the leader's calibrated range (ticks), the sim angle at both ends and at
the calibration pose (should be 0), and the sim model's limits. A range end beyond the
sim limit is clamped there; that is marked with *.
"""

import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "xle_leisaac", "teleop"))
from leader_arm import HOMED_TICKS, JOINTS, LeaderToSim, load_calibration, normalize  # noqa: E402


def urdf_limits(path):
    limits = {}
    for joint in ET.parse(path).getroot().iter("joint"):
        limit = joint.find("limit")
        if joint.get("name") in JOINTS and limit is not None:
            limits[joint.get("name")] = (float(limit.get("lower")), float(limit.get("upper")))
    return limits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--leader", default="calibration/my_leader.json")
    p.add_argument("--follower", default="calibration/my_follower.json")
    args = p.parse_args()

    leader = load_calibration(os.path.join(REPO, args.leader))
    follower = load_calibration(os.path.join(REPO, args.follower))
    urdf = os.path.join(os.environ["XLE_THIRD_PARTY"], "so_arm100_urdf/so101_new_calib_camera.urdf")
    limits = urdf_limits(urdf)
    to_sim = LeaderToSim(leader, follower, limits)

    print(f"leader {args.leader}, follower {args.follower}")
    print(f"{'joint':14s} {'leader ticks':>13s} {'sim at min':>10s} {'at calib':>9s} {'at max':>8s}   sim limits (deg)")
    ok = True
    for i, joint in enumerate(JOINTS):
        c = leader[joint]
        angles = []
        for ticks in (c["range_min"], HOMED_TICKS, c["range_max"]):
            sample = [HOMED_TICKS] * 6
            sample[i] = ticks
            angles.append(math.degrees(to_sim(sample)[i]))
        lo, hi = (math.degrees(v) for v in limits[joint])
        raw = [(t - HOMED_TICKS) * 360 / 4096 for t in (c["range_min"], c["range_max"])]
        mark = ["*" if joint != "gripper" and not lo - 0.01 <= r <= hi + 0.01 else " " for r in raw]
        print(f"{joint:14s} {c['range_min']:5d}..{c['range_max']:<6d} {angles[0]:+9.1f}{mark[0]} {angles[1]:+8.1f}"
              f"  {angles[2]:+7.1f}{mark[1]}  [{lo:+.0f}, {hi:+.0f}]")
        if joint != "gripper":
            # Same calibration on both sides: the follower must copy the leader's angle exactly.
            for ticks in range(c["range_min"], c["range_max"] + 1, 37):
                sample = [HOMED_TICKS] * 6
                sample[i] = ticks
                want = min(hi, max(lo, (ticks - HOMED_TICKS) * 360 / 4096))
                ok &= abs(math.degrees(to_sim(sample)[i]) - want) < 1e-6
        else:
            ok &= abs(angles[0] - lo) < 1e-6 and abs(angles[2] - hi) < 1e-6
        ok &= abs(normalize(joint, c["range_min"], leader) - (0 if joint == "gripper" else -100)) < 1e-9
    if leader["gripper"]["range_max"] - leader["gripper"]["range_min"] < 100:
        print("WARNING: gripper range is under 100 ticks; was the trigger moved during calibration?")
        ok = False
    print("* = beyond the sim limit, clamped there")
    print(f"[check] leader mapping {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
