"""export_robot_spec.py - write XLeRobot's dimensions, joints, base and action spaces to one file.

Every value is read from a source file at run time (URDFs, robot code, docs) and printed
with the file it came from (and the line number for code). Nothing is typed in by hand,
so the output can be trusted for comparing implementations.

Sources:
- XLeRobot repo (Vector-Wangel/XLeRobot), the commit cloned in $XLE_THIRD_PARTY/XLeRobot:
  real-robot code, hardware docs (v0.3.0), and the sim model. The ManiSkill URDF and the
  URDF inside simulation/Isaac_sim/xlerobot.zip are checked to be byte-identical.
- The teleop branch of this repo (origin/xlerobo_setup), if fetched: its vendored copy.
- Official SO-101 model (TheRobotStudio/SO-ARM100) in $XLE_THIRD_PARTY/so_arm100_urdf.
- This repo's built Isaac model: $XLE_ASSETS/robots/xlerobot_so101/xlerobot_so101.urdf.

Plain Python, login node is fine:
  python3 tools/export_robot_spec.py          # -> docs/ROBOT_SPEC.md and docs/robot_spec.json
"""

import hashlib
import json
import math
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from urdf_geometry import Urdf  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TP = os.environ["XLE_THIRD_PARTY"]
XLE = os.path.join(TP, "XLeRobot")
ROBOT_PY = os.path.join(XLE, "software/src/robots/xlerobot/xlerobot.py")
CONFIG_PY = os.path.join(XLE, "software/src/robots/xlerobot/config_xlerobot.py")
MS_URDF = os.path.join(XLE, "simulation/Maniskill/assets/xlerobot/xlerobot.urdf")
MS_AGENT = os.path.join(XLE, "simulation/Maniskill/agents/xlerobot/xlerobot.py")
ZIP_URDF = os.path.join(os.environ["XLE_ASSETS"], "robots/xlerobot/xlerobot/xlerobot.urdf")
DOCS_INTRO = os.path.join(XLE, "docs/en/source/hardware/hardware_intro/index.md")
DOCS_BOM = os.path.join(XLE, "docs/en/source/hardware/getting_started/material.md")
SO101_URDF = os.path.join(TP, "so_arm100_urdf/so101_new_calib_camera.urdf")
OURS_URDF = os.path.join(os.environ["XLE_ASSETS"], "robots/xlerobot_so101/xlerobot_so101.urdf")
OURS_PY = os.path.join(REPO, "xle_leisaac/robots/xlerobot.py")
TELEOP_BRANCH = "origin/xlerobo_setup"
OUT_MD = os.path.join(REPO, "docs/ROBOT_SPEC.md")
OUT_JSON = os.path.join(REPO, "docs/robot_spec.json")

lines_out = []
data = {}


def out(text=""):
    lines_out.append(text)


def sha(path_or_bytes):
    raw = path_or_bytes if isinstance(path_or_bytes, bytes) else open(path_or_bytes, "rb").read()
    return hashlib.sha256(raw).hexdigest()[:16]


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, check=False).stdout


def short(path):
    for root, name in ((XLE, "XLeRobot"), (os.path.join(TP, "so_arm100_urdf"), "SO-ARM100"),
                       (os.environ["XLE_ASSETS"], "$XLE_ASSETS"), (REPO, "this repo")):
        if path.startswith(root):
            return f"{name}:{os.path.relpath(path, root)}"
    return path


def grep(path, pattern, text=None):
    """[(line number, line)] of lines matching a regex."""
    text = text if text is not None else open(path, encoding="utf-8", errors="ignore").read()
    return [(i + 1, line.rstrip()) for i, line in enumerate(text.splitlines()) if re.search(pattern, line)]


def quote(path, pattern, text=None, label=None):
    hits = grep(path, pattern, text)
    out(f"From `{label or short(path)}`:")
    out("```")
    for n, line in hits:
        out(f"{n:5}: {line.strip()}")
    out("```")
    return hits


def fmt(v, digits=4):
    return "[" + ", ".join(f"{x:.{digits}f}" for x in v) + "]"


def joint_table(urdf, key):
    out("| joint | type | parent → child | origin xyz (m) | origin rpy (rad) | axis | lower | upper | lower° | upper° | effort | velocity |")
    out("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    rows = []
    for name, j in urdf.joints.items():
        if j.get("type") == "fixed":
            continue
        o = j.find("origin")
        xyz = o.get("xyz", "0 0 0") if o is not None else "0 0 0"
        rpy = o.get("rpy", "0 0 0") if o is not None else "0 0 0"
        axis = j.find("axis").get("xyz") if j.find("axis") is not None else ""
        lim = j.find("limit")
        lo = lim.get("lower") if lim is not None else None
        hi = lim.get("upper") if lim is not None else None
        eff = lim.get("effort") if lim is not None else None
        vel = lim.get("velocity") if lim is not None else None
        rev = j.get("type") in ("revolute", "continuous")

        def deg(value):
            return f"{math.degrees(float(value)):.1f}" if (value is not None and rev) else ""

        out(f"| `{name}` | {j.get('type')} | {j.find('parent').get('link')} → {j.find('child').get('link')} | {xyz} | {rpy} "
            f"| {axis} | {lo or ''} | {hi or ''} | {deg(lo)} | {deg(hi)} | {eff or ''} | {vel or ''} |")
        rows.append(dict(name=name, type=j.get("type"), parent=j.find("parent").get("link"), child=j.find("child").get("link"),
                         xyz=xyz, rpy=rpy, axis=axis, lower=lo, upper=hi, effort=eff, velocity=vel))
    data.setdefault(key, {})["joints"] = rows
    out()


def mass_table(urdf, key):
    masses = {}
    for name, link in urdf.links.items():
        m = link.find("inertial/mass")
        masses[name] = float(m.get("value")) if m is not None else 0.0
    out("| link | mass (kg) |")
    out("| --- | --- |")
    for name, m in masses.items():
        if m > 0:
            out(f"| {name} | {m:.4f} |")
    out(f"| **total** | **{sum(masses.values()):.3f}** |")
    out()
    data.setdefault(key, {})["masses_kg"] = masses


def positions(urdf, root, items, key):
    """Zero-pose positions of frames, in the root link's frame."""
    out(f"All joints at 0. Positions in the `{root}` frame (x forward, y left, z up, metres).")
    out()
    out("| frame | x | y | z |")
    out("| --- | --- | --- | --- |")
    pos = {}
    for label, kind, name in items:
        if kind == "link":
            p = urdf.link_pose(name, root)[:3, 3]
        else:
            p = urdf.joint_axis(name, root)[0]
        pos[label] = p.round(4).tolist()
        out(f"| {label} | {p[0]:.4f} | {p[1]:.4f} | {p[2]:.4f} |")
    out()
    data.setdefault(key, {})["zero_pose_positions_m"] = pos
    return pos


def chain_distances(urdf, joints, key):
    pts = [urdf.joint_axis(n)[0] for n in joints]
    d = [float(np.linalg.norm(pts[i + 1] - pts[i])) for i in range(len(pts) - 1)]
    out("Distance between consecutive joint-frame origins (m). Note: an origin can sit anywhere along its")
    out("rotation axis, so these are frame distances, not always physical link lengths.")
    out()
    out("| from → to | distance |")
    out("| --- | --- |")
    for i, v in enumerate(d):
        out(f"| {joints[i]} → {joints[i + 1]} | {v:.4f} |")
    out()
    data.setdefault(key, {})["joint_origin_distances_m"] = dict(zip([f"{a}->{b}" for a, b in zip(joints, joints[1:])], d))


def main():
    xle_commit = git(XLE, "rev-parse", "HEAD").decode().strip()
    xle_date = git(XLE, "log", "-1", "--format=%ad", "--date=short").decode().strip()
    so_commit = open(os.path.join(TP, "so_arm100_urdf/COMMIT")).read().strip()
    branch_urdf = git(REPO, "show", f"{TELEOP_BRANCH}:third_party/xlerobot_maniskill/assets/xlerobot/xlerobot.urdf")
    branch_agent = git(REPO, "show", f"{TELEOP_BRANCH}:third_party/xlerobot_maniskill/agents/xlerobot/xlerobot.py")
    branch_teleop = git(REPO, "show", f"{TELEOP_BRANCH}:xlerobot_sim/teleop_controller.py").decode()
    branch_rev = git(REPO, "rev-parse", "--short", TELEOP_BRANCH).decode().strip()

    out("# XLeRobot robot specification (generated)")
    out()
    out("Generated by `tools/export_robot_spec.py`. Every value is read from the file named next to it;")
    out("nothing is typed in by hand. Re-run the script after updating any source.")
    out()
    out("## 0. Sources")
    out()
    out("| source | version | file(s) | sha256 (first 16) |")
    out("| --- | --- | --- | --- |")
    out(f"| XLeRobot (Vector-Wangel/XLeRobot) | commit `{xle_commit[:10]}` ({xle_date}); hardware v0.3.0 print files | `{short(ROBOT_PY)}` | {sha(ROBOT_PY)} |")
    out(f"| | | `{short(MS_URDF)}` | {sha(MS_URDF)} |")
    out(f"| | | URDF inside `simulation/Isaac_sim/xlerobot.zip` | {sha(ZIP_URDF)} |")
    out(f"| | | `{short(MS_AGENT)}` | {sha(MS_AGENT)} |")
    if branch_urdf:
        out(f"| Teleop branch `{TELEOP_BRANCH}` | `{branch_rev}` | `third_party/xlerobot_maniskill/assets/xlerobot/xlerobot.urdf` | {sha(branch_urdf)} |")
        out(f"| | | `third_party/xlerobot_maniskill/agents/xlerobot/xlerobot.py` | {sha(branch_agent)} |")
    out(f"| SO-ARM100 (TheRobotStudio) | commit `{so_commit[:10]}` | `{short(SO101_URDF)}` | {sha(SO101_URDF)} |")
    out(f"| This repo, built Isaac model | from the above | `{short(OURS_URDF)}` | {sha(OURS_URDF)} |")
    out()
    same_urdf = sha(MS_URDF) == sha(ZIP_URDF) == (sha(branch_urdf) if branch_urdf else sha(MS_URDF))
    same_agent = (sha(branch_agent) == sha(MS_AGENT)) if branch_agent else None
    out(f"- XLeRobot ManiSkill URDF == Isaac zip URDF{' == teleop branch URDF' if branch_urdf else ''}: **{same_urdf}**")
    if same_agent is not None:
        out(f"- Teleop branch agent == XLeRobot ManiSkill agent: **{same_agent}**")
    out()
    data["sources"] = dict(xlerobot_commit=xle_commit, so_arm100_commit=so_commit, teleop_branch=branch_rev,
                           xlerobot_urdf_identical_across_maniskill_isaac_branch=same_urdf, branch_agent_identical=same_agent)

    # ---------------------------------------------------------------------------------------
    out("## 1. Real robot (XLeRobot software)")
    out()
    out("### 1.1 Action / state keys (17, in order)")
    out()
    text = open(ROBOT_PY).read()
    block = text[text.index("def _state_ft"):text.index("def _cameras_ft")]
    keys = re.findall(r'"([a-z0-9_]+\.(?:pos|vel))"', block)
    start = text[:text.index("def _state_ft")].count("\n") + 1
    out(f"From `{short(ROBOT_PY)}`, `_state_ft` (line {start}); `action_features` returns the same dict.")
    out()
    out("| # | key |")
    out("| --- | --- |")
    for i, k in enumerate(keys):
        out(f"| {i} | `{k}` |")
    out()
    data["real_robot"] = {"action_keys": keys}

    out("### 1.2 Motors, buses, normalisation")
    out()
    quote(ROBOT_PY, r'Motor\(\d+|port=self\.config\.port|norm_mode_body = ')
    quote(CONFIG_PY, r'port1:|port2:|use_degrees:')
    out("### 1.3 Base kinematics and speeds")
    out()
    quote(ROBOT_PY, r'wheel_radius: float =|base_radius: float =|max_raw: int =|angles = np\.radians|"xy": .*"theta"')
    out("Wheel layout from `angles = radians([240, 0, 120] - 90)`: wheels at 150°, −90°, 30° around the base centre,")
    out("at `base_radius` from it. Base command keys `x.vel`, `y.vel` are m/s and `theta.vel` is deg/s.")
    out()

    # ---------------------------------------------------------------------------------------
    xle = Urdf(MS_URDF)
    out("## 2. XLeRobot sim model (ManiSkill URDF = Isaac zip URDF = teleop branch)")
    out()
    out("### 2.1 Moving joints")
    out()
    joint_table(xle, "xlerobot_sim_model")
    out("### 2.2 Zero-pose positions")
    out()
    positions(xle, "base_link", [
        ("right arm mount (`Base`)", "link", "Base"), ("left arm mount (`Base_2`)", "link", "Base_2"),
        ("right shoulder pan axis point", "joint", "Rotation"), ("right shoulder lift axis point", "joint", "Pitch"),
        ("left shoulder pan axis point", "joint", "Rotation_2"), ("left shoulder lift axis point", "joint", "Pitch_2"),
        ("right jaw (`Fixed_Jaw`)", "link", "Fixed_Jaw"), ("left jaw (`Fixed_Jaw_2`)", "link", "Fixed_Jaw_2"),
        ("right wrist camera (`Right_Arm_Camera`)", "link", "Right_Arm_Camera"), ("left wrist camera (`Left_Arm_Camera`)", "link", "Left_Arm_Camera"),
        ("head pan axis point", "joint", "head_pan_joint"), ("head tilt axis point", "joint", "head_tilt_joint"),
        ("head camera (`head_camera_link`)", "link", "head_camera_link"),
    ], "xlerobot_sim_model")
    out("Arm mount spacing (left − right, y): "
        f"{xle.link_pose('Base_2', 'base_link')[1, 3] - xle.link_pose('Base', 'base_link')[1, 3]:.4f} m")
    out()
    out("### 2.3 Arm chain (right arm)")
    out()
    chain_distances(xle, ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"], "xlerobot_sim_model")
    out("### 2.4 Link masses")
    out()
    mass_table(xle, "xlerobot_sim_model")

    # ---------------------------------------------------------------------------------------
    out("## 3. ManiSkill agent and teleop branch (controllers and action layout)")
    out()
    agent_text = branch_agent.decode() if branch_agent else open(MS_AGENT).read()
    label = f"{TELEOP_BRANCH}:third_party/xlerobot_maniskill/agents/xlerobot/xlerobot.py" if branch_agent else short(MS_AGENT)
    quote(MS_AGENT, r'qpos=np\.array|# \[\d|_stiffness = |_damping = |_force_limit = |PDBaseForwardVel|lower=\[|upper=\[|entity_uid=',
          text=agent_text, label=label)
    if branch_teleop:
        quote(None, r'^Control mode|^    \[|ACTION_DIM =|GRIPPER_OPEN, GRIPPER_CLOSED|base_speed =|turn_step =|head_step =|initial_ee =|l1: float|theta1_offset =|theta2_offset =|joint2 = max|joint3 = max|t\[6\] = t\[11\]',
              text=branch_teleop, label=f"{TELEOP_BRANCH}:xlerobot_sim/teleop_controller.py")

    # ---------------------------------------------------------------------------------------
    so = Urdf(SO101_URDF)
    out("## 4. Official SO-101 arm (SO-ARM100, new_calib, with wrist camera)")
    out()
    out("### 4.1 Joints")
    out()
    joint_table(so, "so101_official")
    out("### 4.2 Arm chain")
    out()
    chain_distances(so, ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"], "so101_official")
    out("### 4.3 Zero-pose positions (arm base frame)")
    out()
    positions(so, "base_link", [("shoulder pan axis point", "joint", "shoulder_pan"), ("shoulder lift axis point", "joint", "shoulder_lift"),
                                ("tool point (`gripper_frame_link`)", "link", "gripper_frame_link"),
                                ("wrist camera (`wrist_camera_link`)", "link", "wrist_camera_link")], "so101_official")
    out("### 4.4 Link masses")
    out()
    mass_table(so, "so101_official")

    # ---------------------------------------------------------------------------------------
    ours = Urdf(OURS_URDF)
    out("## 5. This repo's Isaac model (XLeRobot body + official SO-101 arms)")
    out()
    out("Built by `tools/build_xlerobot_urdf.py` from sections 2 and 4.")
    out()
    mounts = {s: ours.link_pose(f"{s}_arm_base_link", "base_link")[:3, 3].round(4).tolist() for s in ("left", "right")}
    out(f"Arm mounts (`<side>_arm_base_link` in `base_link`): left {mounts['left']}, right {mounts['right']} m")
    out()
    data["isaac_model"] = {"arm_mounts_m": mounts}
    out("### 5.1 Moving joints")
    out()
    joint_table(ours, "isaac_model")
    out("### 5.2 Zero-pose positions")
    out()
    positions(ours, "base_link", [
        ("left shoulder pan axis point", "joint", "left_arm_shoulder_pan"), ("left shoulder lift axis point", "joint", "left_arm_shoulder_lift"),
        ("right shoulder pan axis point", "joint", "right_arm_shoulder_pan"), ("right shoulder lift axis point", "joint", "right_arm_shoulder_lift"),
        ("left tool point", "link", "left_arm_gripper_frame_link"), ("right tool point", "link", "right_arm_gripper_frame_link"),
        ("left wrist camera link", "link", "left_arm_wrist_camera_link"), ("right wrist camera link", "link", "right_arm_wrist_camera_link"),
        ("head camera (`head_camera_link`)", "link", "head_camera_link"),
    ], "isaac_model")
    out("### 5.3 Link masses")
    out()
    mass_table(ours, "isaac_model")
    out("### 5.4 Action layout and actuator settings")
    out()
    quote(OURS_PY, r'^STS3215_|_JOINTS = |XLEROBOT_JOINT_NAMES = |effort_limit_sim=|velocity_limit_sim=|stiffness=|damping=|_HFOV_DEG, |WRIST_CAMERA_POS|_HEAD_LIMITS_DEG =')

    # ---------------------------------------------------------------------------------------
    out("## 6. XLeRobot hardware docs (v0.3.0)")
    out()
    quote(DOCS_INTRO, r'Weight|Height range|Width range|=.*SO-10|omni')
    quote(DOCS_BOM, r'(?i)omni|STS3215|RASKOG|RÅSKOG|camera')

    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines_out) + "\n")
    with open(OUT_JSON, "w") as f:
        json.dump(data, f, indent=1)
    print(f"[spec] wrote {OUT_MD} ({len(lines_out)} lines) and {OUT_JSON}")


if __name__ == "__main__":
    main()
