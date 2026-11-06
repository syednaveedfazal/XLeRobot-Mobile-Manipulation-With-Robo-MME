"""inspect_usd.py - print what is inside a robot USD file, without simulating it.

Shows the stage units, articulation root, joints (type, limits, parent, child),
rigid bodies with their masses, cameras, and how many meshes are present.
Run it on a robot before writing a config for it, so the joint names are facts.

  slurm/in_container.sh /isaac-sim/python.sh tools/inspect_usd.py <file.usd> [more.usd]
"""

import sys

from isaaclab.app import AppLauncher

files = sys.argv[1:]
simulation_app = AppLauncher(headless=True).app

from pxr import Usd, UsdGeom, UsdPhysics


def main():
    for path in files:
        stage = Usd.Stage.Open(path)
        print(f"\n=== {path}")
        print(f"default prim {stage.GetDefaultPrim().GetPath()}  up {UsdGeom.GetStageUpAxis(stage)}"
              f"  metres/unit {UsdGeom.GetStageMetersPerUnit(stage)}")

        # TraverseInstanceProxies also walks instanced geometry, which a plain
        # Traverse() skips. Robots imported from URDF usually use instancing.
        prims = list(Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()))
        meshes = sum(p.IsA(UsdGeom.Mesh) for p in prims)
        collisions = sum(p.HasAPI(UsdPhysics.CollisionAPI) for p in prims)
        print(f"meshes {meshes}  collision prims {collisions}")

        print("articulation roots:", [str(p.GetPath()) for p in prims if p.HasAPI(UsdPhysics.ArticulationRootAPI)])
        print("cameras:", [str(p.GetPath()) for p in prims if p.IsA(UsdGeom.Camera)])

        print("rigid bodies (mass in kg):")
        for p in prims:
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                mass = p.GetAttribute("physics:mass").Get()
                print(f"  {p.GetName():28} {mass}")

        print("moving joints (limits in degrees or metres):")
        for p in prims:
            if p.IsA(UsdPhysics.Joint) and not p.IsA(UsdPhysics.FixedJoint):
                joint = UsdPhysics.Joint(p)
                parent = [t.name for t in joint.GetBody0Rel().GetTargets()]
                child = [t.name for t in joint.GetBody1Rel().GetTargets()]
                low = p.GetAttribute("physics:lowerLimit").Get()
                high = p.GetAttribute("physics:upperLimit").Get()
                kind = p.GetTypeName().replace("Physics", "").replace("Joint", "")
                print(f"  {p.GetName():24} {kind:10} [{low}, {high}]  {parent} -> {child}")


if __name__ == "__main__":
    main()
    simulation_app.close()
