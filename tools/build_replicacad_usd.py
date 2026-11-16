"""build_replicacad_usd.py - convert a ReplicaCAD apartment (Habitat format) to USD for Isaac Lab.

Plain Python (trimesh + usd-core), no Isaac Sim, so it runs on the login node in a minute or two:

  bash setup/06_replicacad.sh              # apt_0
  bash setup/06_replicacad.sh apt_3        # another furniture layout of the same apartment

A scene is written as three layers, so the apartment can later be swapped for a scan of the
lab without touching the robot or task code:

  shell         scenes/<scene>/shell.usd    walls, floor, fixed furniture, ceiling lights. Static collision.
  objects       objects/<template>.usd      one rigid body per movable object (bowls, books, ...)
  furniture     articulated/<template>.usd  fixed-base articulations: kitchen counter drawers,
                                            fridge doors, cupboard doors, chest of drawers, cabinet
  manifest      scenes/<scene>/manifest.json   what to spawn where, the robot's start pose, and
                                            named "anchors" (counter tops, drawer insides) for tasks

Shared pieces:
  meshes/<name>.usd     visual geometry + materials of one .glb (referenced by the files above)
  textures/<hash>.png   textures, de-duplicated by content

Frames. ReplicaCAD is Y-up, Isaac is Z-up. Every mesh is rotated by C = Rx(+90 deg) when it is
written, and every pose (object poses, URDF joint and collision origins) is conjugated,
T' = C T C^-1, so that all local frames are Z-up too. World positions come out the same as in
ManiSkill's ReplicaCAD scene builder (which applies C on the left instead).

Choices that follow ManiSkill's scene builder:
- doors (door1/2/3, doubledoor) are left out;
- STATIC objects collide with their visual triangle mesh, DYNAMIC ones with ReplicaCAD's convex
  decomposition (or the convex hull of the visual mesh where the config asks for a bounding box).
"""

import argparse
import hashlib
import io
import json
import math
import os
import re
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt

C = trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0])  # Y-up -> Z-up
C_INV = C.T

# Plain-looking materials: glTF roughness 0 (many ReplicaCAD objects) renders as a mirror in RTX.
MIN_ROUGHNESS = 0.35

# ManiSkill places XLeRobot (and Fetch) here, facing +x. Z-up world. The height is set at build
# time to just above the floor: XLeRobot's root is fixed (its base moves on virtual joints), so a
# robot that starts touching the floor mesh is pinned there by contact friction and cannot drive.
ROBOT_SPAWN = {"pos": [-1.0, 0.0, 0.0], "yaw_deg": 0.0}
ROBOT_FLOOR_CLEARANCE = 0.01

# Ceiling lights, Z-up world. Positions from ManiSkill's ReplicaCAD scene builder.
CEILING_LIGHTS = {
    "entrance": [-1.1, 2.775, 2.3],
    "dining": [-0.5, -1.44, 2.3],
    "dining_back": [2.4, -1.6, 2.3],
    "living": [2.5, -6.1, 2.3],
    "stair": [3.14, 3.24, 3.0],
}


# --- small helpers ---------------------------------------------------------------------------
def safe_name(name: str) -> str:
    """A valid USD prim name."""
    name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    return name if name and not name[0].isdigit() else f"_{name}"


def short_name(template: str) -> str:
    """'objects/frl_apartment_bowl_01' -> 'bowl_01'."""
    return os.path.basename(template).replace("frl_apartment_", "")


def quat_wxyz_to_matrix(q) -> np.ndarray:
    return trimesh.transformations.quaternion_matrix(q)  # trimesh uses (w, x, y, z)


def pose_matrix(pos, quat_wxyz) -> np.ndarray:
    T = quat_wxyz_to_matrix(quat_wxyz)
    T[:3, 3] = pos
    return T


def urdf_origin(elem) -> np.ndarray:
    """4x4 from a URDF <origin xyz rpy> (missing -> identity)."""
    if elem is None:
        return np.eye(4)
    xyz = [float(v) for v in elem.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in elem.get("rpy", "0 0 0").split()]
    T = trimesh.transformations.euler_matrix(*rpy, axes="sxyz")
    T[:3, 3] = xyz
    return T


def conj(T: np.ndarray) -> np.ndarray:
    """Express a Y-up transform in the Z-up frame."""
    return C @ T @ C_INV


def matrix_to_pos_quat(T: np.ndarray):
    """4x4 -> (pos list, quat wxyz list)."""
    q = trimesh.transformations.quaternion_from_matrix(T)
    if q[0] < 0:
        q = -q
    return [float(v) for v in T[:3, 3]], [float(v) for v in q]


def set_xform(prim, T: np.ndarray, scale=None):
    """Author translate + orient (+ scale) ops from a 4x4 rigid transform."""
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    pos, q = matrix_to_pos_quat(T)
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*pos))
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(q[0], Gf.Vec3d(*q[1:])))
    if scale is not None:
        xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*[float(s) for s in scale]))


def new_stage(path: str, root: str, kind=UsdGeom.Xform):
    if os.path.exists(path):
        os.remove(path)
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    prim = kind.Define(stage, f"/{root}").GetPrim()
    stage.SetDefaultPrim(prim)
    return stage, prim


def rel(from_file: str, to_file: str) -> str:
    """Relative asset path, as USD wants it (./ or ../ prefix)."""
    r = os.path.relpath(to_file, os.path.dirname(from_file))
    return r if r.startswith("..") else f"./{r}"


# --- meshes ----------------------------------------------------------------------------------
class MeshLibrary:
    """Converts each .glb once into meshes/<name>.usd (visual geometry + materials, Z-up)."""

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.mesh_dir = os.path.join(out_dir, "meshes")
        self.tex_dir = os.path.join(out_dir, "textures")
        os.makedirs(self.mesh_dir, exist_ok=True)
        os.makedirs(self.tex_dir, exist_ok=True)
        self.done: dict[str, str] = {}
        self.stats = {"meshes": 0, "textures": 0, "untextured_materials": 0, "color_visuals": 0}

    def visual(self, glb: str) -> str:
        """Path of the visual USD for this .glb (converted on first use)."""
        if glb in self.done:
            return self.done[glb]
        stem = os.path.splitext(os.path.basename(glb))[0]
        parent = os.path.basename(os.path.dirname(glb))
        name = stem if parent in ("objects", "stages") else f"{parent}__{stem}"  # urdf meshes: folder prefix
        out = os.path.join(self.mesh_dir, f"{safe_name(name)}.usd")
        self._write_visual(glb, out)
        self.done[glb] = out
        return out

    def _texture(self, image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        data = buf.getvalue()
        path = os.path.join(self.tex_dir, hashlib.sha1(data).hexdigest()[:16] + ".png")
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(data)
            self.stats["textures"] += 1
        return path

    def _material(self, stage, looks_path: str, mat, out_file: str):
        """UsdPreviewSurface from a trimesh PBRMaterial / SimpleMaterial."""
        name = safe_name(getattr(mat, "name", None) or "material")
        path = f"{looks_path}/{name}"
        i = 1
        while stage.GetPrimAtPath(path):
            path, i = f"{looks_path}/{name}_{i}", i + 1
        material = UsdShade.Material.Define(stage, path)
        surface = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
        surface.CreateIdAttr("UsdPreviewSurface")

        if isinstance(mat, trimesh.visual.material.PBRMaterial):
            factor = mat.baseColorFactor
            factor = np.ones(4) if factor is None else np.asarray(factor, dtype=float) / (255.0 if np.max(factor) > 1 else 1.0)
            image = mat.baseColorTexture
            metallic = 0.0 if mat.metallicFactor is None else float(mat.metallicFactor)
            roughness = 1.0 if mat.roughnessFactor is None else float(mat.roughnessFactor)
        else:  # SimpleMaterial
            factor = np.asarray(mat.diffuse, dtype=float) / 255.0
            image = getattr(mat, "image", None)
            metallic, roughness = 0.0, 0.8
        surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(max(roughness, MIN_ROUGHNESS))

        if image is not None:
            st = UsdShade.Shader.Define(stage, f"{path}/st")
            st.CreateIdAttr("UsdPrimvarReader_float2")
            st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            st.CreateOutput("result", Sdf.ValueTypeNames.Float2)
            tex = UsdShade.Shader.Define(stage, f"{path}/base_color")
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(rel(out_file, self._texture(image)))
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
            tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
            tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*[float(v) for v in factor]))
            tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
        else:
            self.stats["untextured_materials"] += 1
            surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(v) for v in factor[:3]]))
        material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
        return material

    def _write_visual(self, glb: str, out: str):
        scene = trimesh.load(glb, force="scene", process=False)
        stage, root = new_stage(out, "mesh")
        looks = f"/mesh/Looks"
        materials = {}
        for i, node in enumerate(scene.graph.nodes_geometry):
            T, geom_name = scene.graph[node]
            m = scene.geometry[geom_name]
            if not isinstance(m, trimesh.Trimesh) or len(m.faces) == 0:
                continue
            M = C @ T
            prim = UsdGeom.Mesh.Define(stage, f"/mesh/{safe_name(str(node))}_{i}")
            write_mesh_geometry(prim, m, M)
            prim.CreateDoubleSidedAttr(True)  # ReplicaCAD materials are double sided

            visual = m.visual
            if isinstance(visual, trimesh.visual.TextureVisuals):
                if visual.uv is not None and len(visual.uv) == len(m.vertices):
                    pv = UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
                        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
                    )
                    pv.Set(Vt.Vec2fArray.FromNumpy(np.asarray(visual.uv, dtype=np.float32)))
                mat = visual.material
                key = id(mat)
                if key not in materials:
                    materials[key] = self._material(stage, looks, mat, out)
                UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(materials[key])
            else:  # vertex/face colours: use their mean as a flat colour
                self.stats["color_visuals"] += 1
                rgb = np.asarray(visual.main_color[:3], dtype=float) / 255.0
                prim.CreateDisplayColorAttr([Gf.Vec3f(*map(float, rgb))])
        stage.GetRootLayer().Save()
        self.stats["meshes"] += 1


def write_mesh_geometry(prim: UsdGeom.Mesh, m: trimesh.Trimesh, M: np.ndarray, normals=True):
    pts = trimesh.transform_points(np.asarray(m.vertices, dtype=np.float64), M).astype(np.float32)
    prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
    prim.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(m.faces), 3, dtype=np.int32)))
    prim.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(m.faces, dtype=np.int32).ravel()))
    prim.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    prim.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, pts.min(0))), Gf.Vec3f(*map(float, pts.max(0)))]))
    if normals:
        N = np.linalg.inv(M[:3, :3]).T
        n = np.asarray(m.vertex_normals, dtype=np.float64) @ N.T
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        prim.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(n.astype(np.float32)))
        prim.SetNormalsInterpolation(UsdGeom.Tokens.vertex)


def add_collision_parts(stage, parent_path: str, glb: str, approximation: str):
    """Invisible collision meshes from a .glb, one prim per mesh in the file (Z-up)."""
    scene = trimesh.load(glb, force="scene", process=False)
    group = UsdGeom.Scope.Define(stage, f"{parent_path}/collisions")
    count = 0
    for i, node in enumerate(scene.graph.nodes_geometry):
        T, geom_name = scene.graph[node]
        m = scene.geometry[geom_name]
        if not isinstance(m, trimesh.Trimesh) or len(m.faces) == 0:
            continue
        prim = UsdGeom.Mesh.Define(stage, f"{group.GetPath()}/part_{i}")
        write_mesh_geometry(prim, m, C @ T, normals=False)
        prim.CreatePurposeAttr(UsdGeom.Tokens.guide)  # not rendered
        UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(prim.GetPrim()).CreateApproximationAttr(approximation)
        count += 1
    return count


def add_box_collision(stage, path: str, T_local_zup: np.ndarray, size_yup):
    """A URDF <box> (size in the Y-up link frame) as an invisible collider."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    sx, sy, sz = size_yup
    set_xform(cube.GetPrim(), T_local_zup, scale=(sx, sz, sy))  # C swaps the y and z extents
    cube.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


# --- movable objects -------------------------------------------------------------------------
def build_object(src: str, out_dir: str, lib: MeshLibrary, template: str) -> dict:
    """objects/<name>.usd: one rigid body, visual mesh + convex collision parts."""
    name = os.path.basename(template)
    cfg_path = os.path.join(src, "configs/objects", f"{name}.object_config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    visual_glb = os.path.normpath(os.path.join(os.path.dirname(cfg_path), cfg["render_asset"]))
    out = os.path.join(out_dir, "objects", f"{safe_name(short_name(name))}.usd")
    stage, root = new_stage(out, "object")
    UsdPhysics.RigidBodyAPI.Apply(root)
    mass = float(cfg.get("mass", 1.0))
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(mass)
    vis = stage.DefinePrim("/object/visual", "Xform")
    vis.GetReferences().AddReference(rel(out, lib.visual(visual_glb)))
    if cfg.get("use_bounding_box_for_collision") or "collision_asset" not in cfg:
        parts = add_collision_parts(stage, "/object", visual_glb, "convexHull")
    else:
        coll_glb = os.path.normpath(os.path.join(os.path.dirname(cfg_path), cfg["collision_asset"]))
        parts = add_collision_parts(stage, "/object", coll_glb, "convexHull")
    stage.GetRootLayer().Save()
    return {"usd": out, "mass": mass, "collision_parts": parts}


# --- articulated furniture -------------------------------------------------------------------
def build_articulated(src: str, out_dir: str, lib: MeshLibrary, template: str, scale: float = 1.0) -> dict:
    """articulated/<template>.usd from ReplicaCAD's URDF: rigid-body links, USD physics joints.

    `scale` (the scene's uniform_scale) is baked in: link and joint positions, meshes, colliders,
    drawer travel and anchors. Scaling at spawn time instead would leave joint limits unscaled.

    The base link carries ArticulationRootAPI; Isaac Lab's fix_root_link adds the joint to the
    world at spawn time. Joint drives have zero gains here; the env config sets damping.
    """
    folder = os.path.join(src, "urdf", template)

    def origin(elem):
        T = urdf_origin(elem)
        T[:3, 3] *= scale
        return T

    urdf = os.path.join(folder, f"{template}.urdf")
    robot = ET.parse(urdf).getroot()
    links = {l.get("name"): l for l in robot.findall("link")}
    joints = robot.findall("joint")
    children = {j.find("child").get("link") for j in joints}
    parent_of = {j.find("child").get("link"): j for j in joints}

    def has_geometry(link):
        return link.find("visual") is not None or link.find("collision") is not None

    # Forward kinematics at q = 0, Y-up asset frame.
    order = [n for n in links if n not in children]  # root first
    world = {order[0]: np.eye(4)}
    queue = list(order)
    while queue:
        name = queue.pop(0)
        for j in joints:
            if j.find("parent").get("link") == name:
                child = j.find("child").get("link")
                world[child] = world[name] @ origin(j.find("origin"))
                queue.append(child)

    # Drop an empty root that only holds the real base through a fixed joint.
    root_name = order[0]
    skip = set()
    if not has_geometry(links[root_name]):
        fixed_children = [j for j in joints if j.find("parent").get("link") == root_name]
        if len(fixed_children) == 1 and fixed_children[0].get("type") == "fixed":
            skip.add(root_name)
            root_name = fixed_children[0].find("child").get("link")

    suffix = "" if abs(scale - 1.0) < 1e-6 else f"_s{scale:g}".replace(".", "p")
    out = os.path.join(out_dir, "articulated", f"{safe_name(template)}{suffix}.usd")
    # The root link is the asset's origin. Isaac Lab places the ROOT LINK at the spawn pose on every
    # reset, so a root link with its own offset (ReplicaCAD's URDFs rotate theirs by 90 degrees)
    # would come out rotated. The offset goes into the instance pose instead (root_offset).
    base_inv = np.linalg.inv(world[root_name])
    stage, top = new_stage(out, safe_name(template))
    top_path = top.GetPath()
    link_paths = {}
    for name, link in links.items():
        if name in skip:
            continue
        path = f"{top_path}/{safe_name(name)}"
        prim = UsdGeom.Xform.Define(stage, path).GetPrim()
        set_xform(prim, conj(base_inv @ world[name]))
        UsdPhysics.RigidBodyAPI.Apply(prim)
        inertial = link.find("inertial")
        mass = float(inertial.find("mass").get("value")) if inertial is not None else 1.0
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
        for k, v in enumerate(link.findall("visual")):
            geom = v.find("geometry/mesh")
            if geom is None:
                continue
            glb = os.path.join(folder, geom.get("filename"))
            sx, sy, sz = [float(s) for s in geom.get("scale", "1 1 1").split()]
            vprim = stage.DefinePrim(f"{path}/visual_{k}", "Xform")
            set_xform(vprim, conj(origin(v.find("origin"))), scale=(sx * scale, sz * scale, sy * scale))
            vprim.GetReferences().AddReference(rel(out, lib.visual(glb)))
        UsdGeom.Scope.Define(stage, f"{path}/collisions")
        for k, c in enumerate(link.findall("collision")):
            box = c.find("geometry/box")
            if box is None:
                raise NotImplementedError(f"{urdf}: only <box> collisions are handled")
            size = [float(s) * scale for s in box.get("size").split()]
            add_box_collision(stage, f"{path}/collisions/box_{k}", conj(origin(c.find("origin"))), size)
        link_paths[name] = path
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(link_paths[root_name]))

    joint_info = []
    UsdGeom.Scope.Define(stage, f"{top_path}/joints")
    for j in joints:
        parent, child = j.find("parent").get("link"), j.find("child").get("link")
        if parent in skip or child in skip:
            continue
        jtype = j.get("type")
        jpath = f"{top_path}/joints/{safe_name(j.get('name'))}"
        O = conj(origin(j.find("origin")))  # joint frame in the parent link frame (Z-up)
        if jtype == "fixed":
            joint = UsdPhysics.FixedJoint.Define(stage, jpath)
            R_axis = np.eye(4)
        else:
            axis = np.array([float(v) for v in (j.find("axis").get("xyz") if j.find("axis") is not None else "1 0 0").split()])
            axis = C[:3, :3] @ (axis / np.linalg.norm(axis))
            # Rotate the joint frame so that its X axis is the joint axis; then use axis "X".
            R_axis = rotation_x_to(axis)
            lim = j.find("limit")
            lower, upper = float(lim.get("lower")), float(lim.get("upper"))
            if jtype == "prismatic":
                lower, upper = lower * scale, upper * scale
            if jtype == "prismatic":
                joint = UsdPhysics.PrismaticJoint.Define(stage, jpath)
                joint.CreateLowerLimitAttr(lower)
                joint.CreateUpperLimitAttr(upper)
                drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
            elif jtype in ("revolute", "continuous"):
                joint = UsdPhysics.RevoluteJoint.Define(stage, jpath)
                if jtype == "revolute":
                    joint.CreateLowerLimitAttr(math.degrees(lower))  # USD angles are degrees
                    joint.CreateUpperLimitAttr(math.degrees(upper))
                drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
            else:
                raise NotImplementedError(f"{urdf}: joint type {jtype}")
            joint.CreateAxisAttr("X")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(0.0)
            drive.CreateDampingAttr(0.0)
            joint_info.append({"name": j.get("name"), "type": jtype, "lower": lower, "upper": upper})
        joint.CreateBody0Rel().SetTargets([link_paths[parent]])
        joint.CreateBody1Rel().SetTargets([link_paths[child]])
        F0 = O @ R_axis
        p0, q0 = matrix_to_pos_quat(F0)
        _, q1 = matrix_to_pos_quat(R_axis)
        joint.CreateLocalPos0Attr(Gf.Vec3f(*p0))
        joint.CreateLocalRot0Attr(Gf.Quatf(q0[0], Gf.Vec3f(*q0[1:])))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        joint.CreateLocalRot1Attr(Gf.Quatf(q1[0], Gf.Vec3f(*q1[1:])))
    stage.GetRootLayer().Save()

    # Named places to put things (Habitat "receptacles"), in their link's Z-up frame.
    anchors = []
    ao_cfg = os.path.join(folder, f"{template}.ao_config.json")
    if os.path.exists(ao_cfg):
        with open(ao_cfg) as f:
            for rec in json.load(f).get("user_defined", {}).values():
                link = rec["parent_link"]
                pos = (C[:3, :3] @ np.array(rec["position"]) * scale).tolist()
                sx, sy, sz = np.abs(rec["scale"]) * scale
                name = rec["name"].replace("receptacle_aabb_", "").replace("_frl_apartment", "")
                name = name.replace(f"_{template}", "")
                anchors.append({"name": name, "link": safe_name(link), "pos": pos, "half_extents": [float(sx), float(sz), float(sy)]})
    return {"usd": out, "joints": joint_info, "anchors": anchors, "root_link": safe_name(root_name),
            "root_offset": conj(world[root_name])}


def rotation_x_to(axis: np.ndarray) -> np.ndarray:
    """4x4 rotation taking +X onto the unit vector `axis`."""
    x = np.array([1.0, 0.0, 0.0])
    v = np.cross(x, axis)
    c = float(np.dot(x, axis))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(4) if c > 0 else trimesh.transformations.rotation_matrix(math.pi, [0, 0, 1])
    return trimesh.transformations.rotation_matrix(math.acos(np.clip(c, -1, 1)), v)


# --- the scene -------------------------------------------------------------------------------
def build_shell(src, out_dir, lib, scene_json, scene_dir) -> tuple[str, list]:
    """shell.usd: the stage mesh, STATIC objects (triangle-mesh colliders) and lights."""
    out = os.path.join(scene_dir, "shell.usd")
    stage, root = new_stage(out, "shell")
    stage_template = os.path.basename(scene_json["stage_instance"]["template_name"])
    stage_glb = os.path.join(src, "stages", f"{stage_template}.glb")
    statics = [("stage", lib.visual(stage_glb), np.eye(4))]
    static_names = []
    for k, obj in enumerate(scene_json["object_instances"]):
        if obj["motion_type"] != "STATIC":
            continue
        name = os.path.basename(obj["template_name"])
        with open(os.path.join(src, "configs/objects", f"{name}.object_config.json")) as f:
            render = json.load(f)["render_asset"]
        glb = os.path.normpath(os.path.join(src, "configs/objects", render))
        T = conj(pose_matrix(obj["translation"], obj["rotation"]))
        pname = safe_name(f"{short_name(name)}_{k}")
        statics.append((pname, lib.visual(glb), T))
        static_names.append(pname)
    for pname, usd, T in statics:
        prim = stage.DefinePrim(f"/shell/{pname}", "Xform")
        set_xform(prim, T)
        prim.GetReferences().AddReference(rel(out, usd))
    # Static triangle-mesh collision on every mesh of the shell.
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")

    # Warm ceiling lights. The sky (a dome light) is global, so it is added by the env config.
    UsdGeom.Scope.Define(stage, "/shell/lights")
    for name, pos in CEILING_LIGHTS.items():
        light = UsdLux.SphereLight.Define(stage, f"/shell/lights/{name}")
        light.CreateRadiusAttr(0.12)
        light.CreateIntensityAttr(30000.0)
        light.CreateColorAttr(Gf.Vec3f(1.0, 0.88, 0.72))
        set_xform(light.GetPrim(), pose_matrix(pos, [1, 0, 0, 0]))
    stage.GetRootLayer().Save()
    return out, static_names


def snap_upright(T: np.ndarray, tolerance_deg=3.0) -> np.ndarray:
    """Make a nearly upright pose exactly upright (a pure turn about z).

    ReplicaCAD's furniture is rotated by 1.565 rad instead of pi/2 and by quaternions written as
    0.707, which leaves it tilted by a fraction of a degree: enough for hinges to be off vertical.
    """
    R = T[:3, :3]
    if math.degrees(math.acos(np.clip(R[2, 2], -1.0, 1.0))) > tolerance_deg:
        return T
    yaw = math.atan2(R[1, 0], R[0, 0])
    out = trimesh.transformations.rotation_matrix(yaw, [0, 0, 1])
    out[:3, 3] = T[:3, 3]
    return out


def info_path(scene_dir: str, relative: str) -> str:
    return os.path.normpath(os.path.join(scene_dir, relative))


def collision_triangles(usd_path: str, M: np.ndarray, guide_only=False) -> np.ndarray:
    """World triangles (n, 3, 3) of every collider (meshes and cubes) in a USD, placed by M."""
    stage = Usd.Stage.Open(usd_path)
    out = []
    unit_cube = trimesh.creation.box(extents=(1, 1, 1))
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        X = M @ np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).T
        if prim.IsA(UsdGeom.Cube):
            size = UsdGeom.Cube(prim).GetSizeAttr().Get()
            pts, faces = unit_cube.vertices * size, unit_cube.faces
        elif prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
            faces = np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        else:
            continue
        out.append(trimesh.transform_points(pts, X)[faces])
    return np.concatenate(out) if out else np.zeros((0, 3, 3))


class Supports:
    """Upward-facing collider surfaces (floor, counter tops, shelves, objects already placed)."""

    def __init__(self):
        self.tris = np.zeros((0, 3, 3))

    def add_triangles(self, tris: np.ndarray):
        n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        up = n[:, 2] > 0.3 * np.linalg.norm(n, axis=1)
        self.tris = np.concatenate([self.tris, tris[up]])

    def add_usd(self, usd_path: str, M: np.ndarray):
        self.add_triangles(collision_triangles(usd_path, M))

    def height(self, x: float, y: float, below: float):
        """Highest support surface at (x, y) that is not above `below`, or None."""
        a, b, c = self.tris[:, 0], self.tris[:, 1], self.tris[:, 2]
        v0, v1 = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2]
        v2 = np.array([x, y]) - a[:, :2]
        d = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
        ok = np.abs(d) > 1e-12
        d = np.where(ok, d, 1.0)
        u = (v2[:, 0] * v1[:, 1] - v2[:, 1] * v1[:, 0]) / d
        v = (v0[:, 0] * v2[:, 1] - v0[:, 1] * v2[:, 0]) / d
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1)
        z = a[hit, 2] + u[hit] * (b[hit, 2] - a[hit, 2]) + v[hit] * (c[hit, 2] - a[hit, 2])
        z = z[z <= below]
        return float(z.max()) if len(z) else None


def place_objects(objects: list, scene_dir: str, supports: Supports, clearance=0.002, depth=0.06):
    """Lift objects that start sunk into the surface they stand on, so PhysX does not shoot them
    out at the first step (ReplicaCAD poses come from another physics engine).

    Only the object's lowest collider points are tested, against surfaces at most `depth` above
    them, so an object next to a table is never lifted onto it. Objects are placed bottom-up and
    each one's collider parts then support the ones above it. Returns [(name, lift in m)].
    """
    local = {}
    for o in objects:
        if o["usd"] not in local:
            local[o["usd"]] = collision_triangles(info_path(scene_dir, o["usd"]), np.eye(4))
    world = [
        trimesh.transform_points(local[o["usd"]].reshape(-1, 3), pose_matrix(o["pos"], o["rot"])).reshape(-1, 3, 3)
        for o in objects
    ]
    order = sorted(range(len(objects)), key=lambda i: world[i][..., 2].min())
    lifted = []
    for i in order:
        o, tris = objects[i], world[i]
        pts = np.unique(tris.reshape(-1, 3).round(5), axis=0)
        bottom, top = pts[:, 2].min(), pts[:, 2].max()
        low = pts[pts[:, 2] <= bottom + 0.25 * (top - bottom)]
        low = low[np.linspace(0, len(low) - 1, min(len(low), 40)).astype(int)]
        sink = 0.0
        for x, y, z in low:
            h = supports.height(x, y, z + depth)
            if h is not None:
                sink = max(sink, h - z)
        if sink > 0.0:
            lift = sink + clearance
            o["pos"][2] += lift
            tris = tris + np.array([0.0, 0.0, lift])
            lifted.append((o["name"], lift))
        supports.add_triangles(tris)
    return lifted


def build_scene(src: str, out_dir: str, scene: str):
    with open(os.path.join(src, "configs/scenes", f"{scene}.scene_instance.json")) as f:
        scene_json = json.load(f)
    scene_dir = os.path.join(out_dir, "scenes", scene)
    os.makedirs(scene_dir, exist_ok=True)
    for sub in ("objects", "articulated"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    lib = MeshLibrary(out_dir)

    shell, static_names = build_shell(src, out_dir, lib, scene_json, scene_dir)
    print(f"[replicacad] shell: stage + {len(static_names)} static objects -> {shell}")

    articulations, counts = [], {}
    for meta in scene_json["articulated_object_instances"]:
        template = meta["template_name"]
        if "door" in template:  # as in ManiSkill: room doors are left out
            continue
        info = build_articulated(src, out_dir, lib, template, float(meta.get("uniform_scale", 1.0)))
        counts[template] = counts.get(template, 0) + 1
        pos, quat = matrix_to_pos_quat(snap_upright(conj(pose_matrix(meta["translation"], meta["rotation"])) @ info["root_offset"]))
        name = safe_name(template if counts[template] == 1 else f"{template}_{counts[template]}")
        articulations.append({
            "name": name,
            "usd": rel(os.path.join(scene_dir, "manifest.json"), info["usd"]),
            "pos": pos, "rot": quat,
            "fixed_base": bool(meta.get("fixed_base", True)),
            "root_link": info["root_link"],
            "joints": info["joints"],
            "anchors": info["anchors"],
        })
        print(f"[replicacad] furniture {name}: {len(info['joints'])} joints, {len(info['anchors'])} anchors")

    supports = Supports()
    supports.add_usd(shell, np.eye(4))
    robot_spawn = dict(ROBOT_SPAWN, pos=list(ROBOT_SPAWN["pos"]))
    floor = supports.height(robot_spawn["pos"][0], robot_spawn["pos"][1], 0.5)
    robot_spawn["pos"][2] = (floor or 0.0) + ROBOT_FLOOR_CLEARANCE
    for a in articulations:
        supports.add_usd(info_path(scene_dir, a["usd"]), pose_matrix(a["pos"], a["rot"]))

    objects, built = [], {}
    for k, obj in enumerate(scene_json["object_instances"]):
        if obj["motion_type"] != "DYNAMIC":
            continue
        template = obj["template_name"]
        if template not in built:
            built[template] = build_object(src, out_dir, lib, template)
        pos, quat = matrix_to_pos_quat(conj(pose_matrix(obj["translation"], obj["rotation"])))
        objects.append({
            "name": safe_name(f"{short_name(template)}_{k}"),
            "usd": rel(os.path.join(scene_dir, "manifest.json"), built[template]["usd"]),
            "pos": pos, "rot": quat, "mass": built[template]["mass"],
        })
    lifted = place_objects(objects, scene_dir, supports)
    print(f"[replicacad] {len(objects)} movable objects ({len(built)} distinct); "
          f"{len(lifted)} lifted out of a surface they started inside: "
          + ", ".join(f"{n} {d * 100:.1f} cm" for n, d in lifted))

    manifest = {
        "source": f"ReplicaCAD {scene}",
        "shell": rel(os.path.join(scene_dir, "manifest.json"), shell),
        "robot_spawn": robot_spawn,
        "objects": objects,
        "articulations": articulations,
        "notes": "Poses are Z-up world, metres, quaternions (w, x, y, z). Anchor poses are in the "
                 "frame of the named link of their articulation; half_extents are along that frame's axes.",
    }
    with open(os.path.join(scene_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"[replicacad] meshes {lib.stats}")
    print(f"[replicacad] wrote {os.path.join(scene_dir, 'manifest.json')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default="apt_0", help="apt_0 .. apt_5")
    parser.add_argument("--src", default=os.path.join(os.environ.get("MS_ASSET_DIR", ""), "data/scene_datasets/replica_cad_dataset"))
    parser.add_argument("--out", default=os.path.join(os.environ.get("XLE_ASSETS", ""), "replicacad"))
    parser.add_argument("--clean", action="store_true", help="delete the output folder first")
    args = parser.parse_args()
    if not os.path.isdir(args.src):
        raise SystemExit(f"ReplicaCAD not found at {args.src} (run setup/install.sh, or pass --src)")
    if args.clean and os.path.isdir(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out, exist_ok=True)
    build_scene(args.src, args.out, args.scene)


if __name__ == "__main__":
    main()
