"""check_workspace.py - compare the sim robot's size and reach with the real XLeRobot.

Reference values:
- XLeRobot hardware docs: arm workspace height ~0.5-1.25 m above the ground, and
  ~0.36 m beyond the cart edge.
- IKEA RÅSKOG cart (IKEA's published size): 35 x 45 x 78 cm.

It measures the cart's bounding box in the USD, then puts both arms into many random
poses inside their joint limits and records where the jaw tips end up.

  sbatch --output=$HOME/xle_ws_%j.out slurm/sim_job.sbatch \
      /isaac-sim/python.sh tools/check_workspace.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--samples", type=int, default=3000)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
simulation_app = AppLauncher(args).app

import os

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.assets import Articulation
from pxr import Usd, UsdGeom

from xle_leisaac.robots.xlerobot import LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS, XLEROBOT_CFG

# The official SO-101 model has a gripper_frame_link at the tool point between the jaws.
TIP_BODIES = ["left_arm_gripper_frame_link", "right_arm_gripper_frame_link"]


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 60, device=args.device))
    robot = Articulation(XLEROBOT_CFG.replace(prim_path="/World/Robot"))
    sim.reset()

    # Cart size: bounding box of the base_link geometry (cart + arm bases + neck base).
    stage = sim.stage
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    box = bbox.ComputeWorldBound(stage.GetPrimAtPath("/World/Robot/base_link")).ComputeAlignedRange()
    low, high = np.array(box.GetMin()), np.array(box.GetMax())
    print(f"[ws] base_link box min {np.round(low, 3)} max {np.round(high, 3)}")
    print(f"[ws] base_link size x={high[0] - low[0]:.3f} y={high[1] - low[1]:.3f} z={high[2] - low[2]:.3f} m"
          "  (RÅSKOG cart: 0.35 x 0.45 x 0.78 m, orientation unknown)")

    # Random arm poses inside the joint limits, same pose for both arms (mirrored model).
    limits = robot.data.joint_pos_limits[0]
    ids = robot.find_joints(LEFT_ARM_JOINTS[:5] + RIGHT_ARM_JOINTS[:5], preserve_order=True)[0]
    tip_ids = robot.find_bodies(TIP_BODIES, preserve_order=True)[0]
    rng = np.random.default_rng(0)
    tips = []
    for _ in range(args.samples):
        pos = robot.data.default_joint_pos.clone()
        low_l, high_l = limits[ids, 0].cpu().numpy(), limits[ids, 1].cpu().numpy()
        pos[0, ids] = torch.tensor(rng.uniform(low_l, high_l), device=sim.device, dtype=pos.dtype)
        robot.write_joint_state_to_sim(pos, torch.zeros_like(pos))
        sim.step(render=False)
        robot.update(sim.get_physics_dt())
        for body in tip_ids:
            tips.append(robot.data.body_pos_w[0, body].cpu().numpy())
    tips = np.array(tips)

    print(f"[ws] tool point height: min {tips[:, 2].min():.3f}  max {tips[:, 2].max():.3f} m  (docs: ~0.5-1.25 m)")
    beyond = {
        "+x (front)": tips[:, 0].max() - high[0],
        "-x (back)": low[0] - tips[:, 0].min(),
        "+y (left)": tips[:, 1].max() - high[1],
        "-y (right)": low[1] - tips[:, 1].min(),
    }
    for side, value in beyond.items():
        print(f"[ws] reach beyond base_link box {side}: {value:.3f} m")
    print("[ws] docs: ~0.36 m beyond the cart edge")
    mass = robot.root_physx_view.get_masses()[0]
    print(f"[ws] total mass {float(mass.sum()):.2f} kg (docs: ~12 kg)")


if __name__ == "__main__":
    main()
    os._exit(0)
