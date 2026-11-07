"""urdf_geometry.py - small helpers to read URDF frames and STL meshes with plain numpy.

Used by build_xlerobot_urdf.py to work out where the official SO-101 arm has to sit on
the XLeRobot body. No Isaac Sim needed.
"""

import math
import os
import struct
import xml.etree.ElementTree as ET

import numpy as np


def transform(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    """4x4 matrix from a URDF origin (rotation order: roll about x, pitch about y, yaw about z)."""
    roll, pitch, yaw = rpy
    c, s = math.cos, math.sin
    rx = np.array([[1, 0, 0], [0, c(roll), -s(roll)], [0, s(roll), c(roll)]])
    ry = np.array([[c(pitch), 0, s(pitch)], [0, 1, 0], [-s(pitch), 0, c(pitch)]])
    rz = np.array([[c(yaw), -s(yaw), 0], [s(yaw), c(yaw), 0], [0, 0, 1]])
    matrix = np.eye(4)
    matrix[:3, :3] = rz @ ry @ rx
    matrix[:3, 3] = xyz
    return matrix


def origin_of(element):
    origin = element.find("origin")
    if origin is None:
        return np.eye(4)
    xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
    return transform(xyz, rpy)


class Urdf:
    def __init__(self, path):
        self.path = path
        self.root = ET.parse(path).getroot()
        self.joints = {j.get("name"): j for j in self.root.findall("joint")}
        self.links = {l.get("name"): l for l in self.root.findall("link")}
        self._joint_to_child = {j.find("child").get("link"): j for j in self.joints.values()}

    def link_pose(self, link, root=None):
        """Pose of a link frame in the root link's frame, with every joint at zero."""
        matrix = np.eye(4)
        while link in self._joint_to_child and link != root:
            joint = self._joint_to_child[link]
            matrix = origin_of(joint) @ matrix
            link = joint.find("parent").get("link")
        return matrix

    def joint_pose(self, name, root=None):
        joint = self.joints[name]
        return self.link_pose(joint.find("parent").get("link"), root) @ origin_of(joint)

    def joint_axis(self, name, root=None):
        """(point, unit direction) of a joint axis in the root frame."""
        pose = self.joint_pose(name, root)
        axis = np.array([float(v) for v in self.joints[name].find("axis").get("xyz").split()])
        direction = pose[:3, :3] @ axis
        return pose[:3, 3], direction / np.linalg.norm(direction)

    def mesh_points(self, link, root=None):
        """All visual mesh vertices of a link, in the root frame."""
        points = []
        for visual in self.links[link].findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            path = os.path.join(os.path.dirname(self.path), mesh.get("filename").replace("package://", ""))
            scale = np.array([float(v) for v in mesh.get("scale", "1 1 1").split()])
            vertices = read_stl(path) * scale
            pose = self.link_pose(link, root) @ origin_of(visual)
            points.append(vertices @ pose[:3, :3].T + pose[:3, 3])
        return np.concatenate(points)


def read_stl(path):
    """Vertices of a binary or ASCII STL file as an (N, 3) array."""
    with open(path, "rb") as f:
        data = f.read()
    count = struct.unpack("<I", data[80:84])[0] if len(data) >= 84 else 0
    if len(data) == 84 + count * 50:  # binary
        records = np.frombuffer(data[84:], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]), count=count)
        return records["v"].reshape(-1, 3).astype(float)
    numbers = [line.split()[1:] for line in data.decode(errors="ignore").splitlines() if line.strip().startswith("vertex")]
    return np.array(numbers, dtype=float)
