"""build_xlerobot_urdf.py - XLeRobot body + two official SO-101 arms, as one URDF.

Why: the arms in XLeRobot's own model are the older SO-100. The real robot uses SO-101
arms (TheRobotStudio/SO-ARM100). This script keeps the XLeRobot body (cart, top base,
neck, head and the virtual base joints) and swaps in the official SO-101 arm, including
the official wrist-camera mount, at the exact place the XLeRobot arms were mounted.

Where the arm goes: both arm bases have the same footprint. The script lines up
  1. the shoulder-pan axes,
  2. the bottom faces of the arm bases (both sit on the same printed plate),
  3. the forward direction (arm reaching along +x at zero pose).
The numbers are computed from the meshes and printed, not typed in by hand.

Mass: the source base_link weighs 74.9 kg, but the XLeRobot hardware docs give ~12 kg for
the whole robot. The base gets whatever is left of 12 kg after the arms (official SO-101
masses) and the head (estimated as SO-101-like links plus a camera).

Joint names become left_arm_shoulder_pan ... right_arm_gripper, which matches the real
robot's action keys. Plain Python, no Isaac Sim needed:

  python3 tools/build_xlerobot_urdf.py
"""

import copy
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from urdf_geometry import Urdf  # noqa: E402

ASSETS = os.environ["XLE_ASSETS"]
XLE_URDF = os.path.join(ASSETS, "robots/xlerobot/xlerobot/xlerobot.urdf")
SO101_URDF = os.path.join(os.environ["XLE_THIRD_PARTY"], "so_arm100_urdf/so101_new_calib_camera.urdf")
OUTPUT = os.path.join(ASSETS, "robots/xlerobot_so101/xlerobot_so101.urdf")

# XLeRobot (SO-100) arm chains to remove: (arm base link, pan joint, side prefix)
XLE_ARMS = {"left_arm_": ("Base_2", "Rotation_2"), "right_arm_": ("Base", "Rotation")}
VIRTUAL_LINKS = ("root", "root_arm_1_link_1", "root_arm_1_link_2")
TOTAL_MASS_KG = 12.0  # XLeRobot hardware docs: "Weight ~12kg"
HEAD_MASSES_KG = {"head_pan_link": 0.25, "head_tilt_link": 0.20}  # estimate


def mount_pose(xle, so101, xle_base, xle_pan):
    """Pose of the SO-101 base_link in the XLeRobot base_link frame."""
    xle_pan_point, xle_pan_dir = xle.joint_axis(xle_pan, "base_link")
    so_pan_point, so_pan_dir = so101.joint_axis("shoulder_pan", "base_link")
    assert abs(abs(xle_pan_dir[2]) - 1) < 1e-3 and abs(abs(so_pan_dir[2]) - 1) < 1e-3, "pan axes must be vertical"
    xle_bottom = xle.mesh_points(xle_base, "base_link")[:, 2].min()
    so_bottom = so101.mesh_points("base_link", "base_link")[:, 2].min()
    # Both models reach along +x at zero pose, so no rotation is needed (checked below).
    offset = np.array([xle_pan_point[0] - so_pan_point[0], xle_pan_point[1] - so_pan_point[1], xle_bottom - so_bottom])
    return offset


def descendants(urdf, link):
    """All links below `link` (inclusive) and the joints connecting them."""
    links, joints, todo = {link}, set(), [link]
    while todo:
        parent = todo.pop()
        for name, joint in urdf.joints.items():
            if joint.find("parent").get("link") == parent:
                joints.add(name)
                child = joint.find("child").get("link")
                links.add(child)
                todo.append(child)
    return links, joints


def absolute_meshes(element, base_dir):
    for mesh in element.iter("mesh"):
        filename = mesh.get("filename").replace("package://", "")
        if not os.path.isabs(filename):
            mesh.set("filename", os.path.normpath(os.path.join(base_dir, filename)))


def main():
    xle, so101 = Urdf(XLE_URDF), Urdf(SO101_URDF)
    robot = copy.deepcopy(xle.root)
    absolute_meshes(robot, os.path.dirname(XLE_URDF))

    # Sanity check: at zero pose both arms reach forward (+x) from their pan axis.
    so_tip = so101.link_pose("gripper_link", "base_link")[:3, 3] - so101.joint_axis("shoulder_pan", "base_link")[0]
    xle_tip = xle.link_pose("Fixed_Jaw", "base_link")[:3, 3] - xle.joint_axis("Rotation", "base_link")[0]
    assert so_tip[0] > 0 and xle_tip[0] > 0, "arms do not both point forward at zero pose"

    for prefix, (xle_base, xle_pan) in XLE_ARMS.items():
        offset = mount_pose(xle, so101, xle_base, xle_pan)
        print(f"[urdf] {prefix}base_link at xyz={np.round(offset, 4).tolist()} in XLeRobot base_link")

        # Remove the old SO-100 arm, including its arm-base link.
        links, joints = descendants(xle, xle_base)
        attach = next(j for j in xle.joints.values() if j.find("child").get("link") == xle_base)
        joints.add(attach.get("name"))
        for element in list(robot):
            if element.tag == "link" and element.get("name") in links:
                robot.remove(element)
            elif element.tag == "joint" and element.get("name") in joints:
                robot.remove(element)

        # Add a prefixed copy of the official SO-101 arm.
        arm = copy.deepcopy(so101.root)
        absolute_meshes(arm, os.path.dirname(SO101_URDF))
        for element in arm:
            if element.tag == "link":
                element.set("name", prefix + element.get("name"))
                robot.append(element)
            elif element.tag == "joint":
                element.set("name", prefix + element.get("name"))
                for tag in ("parent", "child"):
                    element.find(tag).set("link", prefix + element.find(tag).get("link"))
                robot.append(element)
            elif element.tag == "material" and robot.find(f"material[@name='{element.get('name')}']") is None:
                robot.insert(0, element)

        mount = ET.SubElement(robot, "joint", name=f"{prefix}mount", type="fixed")
        ET.SubElement(mount, "origin", xyz=" ".join(f"{v:.5f}" for v in offset), rpy="0 0 0")
        ET.SubElement(mount, "parent", link="base_link")
        ET.SubElement(mount, "child", link=f"{prefix}base_link")

    # Links with no mass (the virtual base links, the head camera frames) would each become
    # 1 kg in PhysX, which treats 0 as "unset". Give them a negligible mass instead.
    for link in robot.findall("link"):
        mass = link.find("inertial/mass")
        if link.get("name") in VIRTUAL_LINKS or mass is None or float(mass.get("value")) <= 0.0:
            inertial = link.find("inertial")
            if inertial is not None:
                link.remove(inertial)
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "mass", value="0.001")
            ET.SubElement(inertial, "inertia", ixx="1e-6", iyy="1e-6", izz="1e-6", ixy="0", ixz="0", iyz="0")

    # Head masses, then give the base whatever is left of the 12 kg total.
    def mass_element(name):
        return robot.find(f"link[@name='{name}']/inertial/mass")

    for name, mass in HEAD_MASSES_KG.items():
        mass_element(name).set("value", str(mass))
    others = sum(
        float(link.find("inertial/mass").get("value"))
        for link in robot.findall("link")
        if link.find("inertial/mass") is not None and link.get("name") != "base_link"
    )
    old = float(mass_element("base_link").get("value"))
    mass_element("base_link").set("value", f"{TOTAL_MASS_KG - others:.4f}")
    print(f"[urdf] base_link mass {old:.3f} -> {TOTAL_MASS_KG - others:.3f} kg (total {TOTAL_MASS_KG} kg)")

    for element in list(robot):  # transmissions refer to the old joint names
        if element.tag == "transmission":
            robot.remove(element)

    robot.set("name", "xlerobot_so101")
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    ET.indent(robot)
    ET.ElementTree(robot).write(OUTPUT, xml_declaration=True, encoding="utf-8")

    check = Urdf(OUTPUT)
    moving = [n for n, j in check.joints.items() if j.get("type") not in ("fixed",)]
    print(f"[urdf] {len(check.links)} links, moving joints: {moving}")
    print(f"[urdf] wrote {OUTPUT}")


if __name__ == "__main__":
    main()
