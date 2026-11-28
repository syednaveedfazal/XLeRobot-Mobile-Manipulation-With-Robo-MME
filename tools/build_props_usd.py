"""build_props_usd.py - make the sandbox's props as small USD files.

Isaac Lab can spawn single boxes, spheres and cylinders, but not these. So this script
builds them out of boxes:

- bin.usd: open-top bin (floor + 4 walls), for putting objects into.
- cup.usd: upside-down cup (top + 4 walls, open bottom) with a 2 x 2 cm knob, 4 cm tall, on
  top that the SO-101 gripper holds. The jaws close like tweezers (a V that meets at the tips),
  so they grip the knob near their tips and are closed only a little past touching it.
  It can hide a 3.5 cm cube underneath, which is what memory tasks need.
- cube.usd: the PutItBack object, a box with a handle: a 6 cm Rubik's-style box (3 x 3
  coloured stickers per face, visual only, on a black core that is the collider) with a
  bright red 2 x 2 x 4 cm handle on top, 80 g, centre of mass low. The box makes it easy to
  see (a 2 cm block was only 2-3 px in pi0.5's 224 px input); the handle is what the SO-101
  holds: its tweezer-like jaws need something about 2 cm across that rises well above their
  rounded tips.
- work_table.usd: static table, 60 x 40 cm top at 70 cm (like the lab's), on four legs.

Each prop's origin is at the centre of its bottom face.

  slurm/in_container.sh /isaac-sim/python.sh tools/build_props_usd.py
"""

import os

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

OUT_DIR = os.path.join(os.environ["XLE_ASSETS"], "props")
WALL = 0.006  # 6 mm walls


def make_prop(path, boxes, mass, color, rigid=True, visuals=(), center_of_mass=None):
    """boxes: list of (name, centre xyz, size xyz[, colour]) in metres, each a collider.

    rigid=False makes a static prop (colliders only, no rigid body). visuals: extra
    (name, centre, size, colour) boxes that are seen but do not collide.
    """
    if os.path.exists(path):
        os.remove(path)  # Stage.CreateNew refuses to overwrite an existing file
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/prop")
    stage.SetDefaultPrim(root.GetPrim())
    if rigid:
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
        mass_api.CreateMassAttr(mass)
        if center_of_mass is not None:
            mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*center_of_mass))
    # Grippy, non-bouncy material, so the SO-101 jaws can hold the props. PhysX averages the two
    # materials' friction by default, and the robot's jaws have the default 0.5; "max" makes the
    # contact as grippy as the prop, like the rubber grip tape usually put on SO-101 jaws.
    material = UsdShade.Material.Define(stage, "/prop/physics_material")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(1.5)
    physics.CreateDynamicFrictionAttr(1.2)
    physics.CreateRestitutionAttr(0.0)
    physx = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx.CreateFrictionCombineModeAttr().Set("max")
    physx.CreateRestitutionCombineModeAttr().Set("min")
    for name, centre, size, *own_color in boxes:
        cube = UsdGeom.Cube.Define(stage, f"/prop/{name}")
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*centre))
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*(own_color[0] if own_color else color))])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material, materialPurpose="physics")
    for name, centre, size, rgb in visuals:
        cube = UsdGeom.Cube.Define(stage, f"/prop/{name}")
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*centre))
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
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


BOX = 0.06  # edge of the box (m); keep in sync with BOX_SIZE in xle_leisaac/tasks/put_it_back_env_cfg.py
HANDLE = (0.02, 0.02, 0.04)  # on top of the box; keep in sync with HANDLE_SIZE there
HANDLE_COLOUR = (0.9, 0.05, 0.05)
STICKER_COLOURS = {  # face normal -> colour, standard Rubik's scheme
    (0, 0, 1): (0.95, 0.95, 0.95), (0, 0, -1): (1.0, 0.85, 0.0), (1, 0, 0): (0.8, 0.05, 0.05),
    (-1, 0, 0): (1.0, 0.45, 0.0), (0, 1, 0): (0.0, 0.3, 0.8), (0, -1, 0): (0.0, 0.6, 0.2),
}


def rubiks_stickers(size, cell=0.02):
    """Thin coloured cell-sized stickers on each face of a box whose bottom centre is the origin."""
    thick, out = 0.0006, []
    for normal, rgb in STICKER_COLOURS.items():
        axis = [i for i in range(3) if normal[i]][0]
        u, v = [i for i in range(3) if i != axis]
        nu, nv = round(size[u] / cell), round(size[v] / cell)
        for a in range(nu):
            for b in range(nv):
                centre = [0.0, 0.0, size[2] / 2]
                centre[axis] += normal[axis] * (size[axis] / 2 + thick / 2)
                centre[u] += (a - (nu - 1) / 2) * cell
                centre[v] += (b - (nv - 1) / 2) * cell
                dims = [cell * 0.86] * 3
                dims[axis] = thick
                out.append((f"sticker_{axis}{normal[axis] + 1}_{a}{b}", tuple(centre), tuple(dims), rgb))
    return out


def work_table(top_z=0.70, length=0.60, depth=0.40, top=0.03, leg=0.04, inset=0.03):
    """Table top at top_z, four legs; the origin is the floor below the top's centre."""
    legs = [
        (f"leg_{i}", (sx * (length / 2 - inset - leg / 2), sy * (depth / 2 - inset - leg / 2), (top_z - top) / 2), (leg, leg, top_z - top))
        for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1)))
    ]
    return [("top", (0, 0, top_z - top / 2), (length, depth, top))] + legs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    make_prop(os.path.join(OUT_DIR, "bin.usd"), open_box(0.14, 0.07, closed_at_top=False), 0.15, (0.2, 0.35, 0.8))
    cup = open_box(0.065, 0.07, closed_at_top=True) + [("knob", (0, 0, 0.09), (0.02, 0.02, 0.04))]
    make_prop(os.path.join(OUT_DIR, "cup.usd"), cup, 0.08, (0.85, 0.55, 0.15))
    box_with_handle = [
        ("core", (0, 0, BOX / 2), (BOX, BOX, BOX)),
        ("handle", (0, 0, BOX + HANDLE[2] / 2), HANDLE, HANDLE_COLOUR),
    ]
    make_prop(os.path.join(OUT_DIR, "cube.usd"), box_with_handle, 0.08, (0.03, 0.03, 0.03),
              visuals=rubiks_stickers((BOX, BOX, BOX)), center_of_mass=(0, 0, 0.025))
    make_prop(os.path.join(OUT_DIR, "work_table.usd"), work_table(), 0.0, (0.55, 0.38, 0.22), rigid=False)


if __name__ == "__main__":
    main()
    simulation_app.close()
