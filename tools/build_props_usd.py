"""build_props_usd.py - make the sandbox's containers as small USD files.

Isaac Lab can spawn single boxes, spheres and cylinders, but not an open box. So this
script builds two props out of thin boxes, each one rigid body with box colliders:

- bin.usd: open-top bin (floor + 4 walls), for putting objects into.
- cup.usd: upside-down cup (top + 4 walls, open bottom) with a 2 cm knob on top that the
  SO-101 gripper can hold. It can hide a 3.5 cm cube underneath, which is what memory
  tasks need.

Each prop's origin is at the centre of its bottom face.

  slurm/in_container.sh /isaac-sim/python.sh tools/build_props_usd.py
"""

import os

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

from pxr import Gf, Usd, UsdGeom, UsdPhysics

OUT_DIR = os.path.join(os.environ["XLE_ASSETS"], "props")
WALL = 0.006  # 6 mm walls


def make_prop(path, boxes, mass, color):
    """boxes: list of (name, centre xyz, size xyz) in metres."""
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/prop")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(mass)
    for name, centre, size in boxes:
        cube = UsdGeom.Cube.Define(stage, f"/prop/{name}")
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*centre))
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    stage.GetRootLayer().Save()
    print(f"[props] wrote {path}")


def open_box(width, height, closed_at_top):
    """Four walls plus either a floor (bin) or a lid (upside-down cup)."""
    half = width / 2
    plate_z = height - WALL / 2 if closed_at_top else WALL / 2
    return [
        ("plate", (0, 0, plate_z), (width, width, WALL)),
        ("wall_px", (half - WALL / 2, 0, height / 2), (WALL, width, height)),
        ("wall_nx", (-half + WALL / 2, 0, height / 2), (WALL, width, height)),
        ("wall_py", (0, half - WALL / 2, height / 2), (width - 2 * WALL, WALL, height)),
        ("wall_ny", (0, -half + WALL / 2, height / 2), (width - 2 * WALL, WALL, height)),
    ]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    make_prop(os.path.join(OUT_DIR, "bin.usd"), open_box(0.14, 0.07, closed_at_top=False), 0.15, (0.2, 0.35, 0.8))
    cup = open_box(0.065, 0.07, closed_at_top=True) + [("knob", (0, 0, 0.08), (0.02, 0.02, 0.02))]
    make_prop(os.path.join(OUT_DIR, "cup.usd"), cup, 0.08, (0.85, 0.55, 0.15))


if __name__ == "__main__":
    main()
    simulation_app.close()
