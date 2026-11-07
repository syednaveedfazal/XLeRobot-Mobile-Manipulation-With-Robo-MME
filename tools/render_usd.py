"""render_usd.py - look at a robot USD from four sides on a plain floor.

Headless jobs have no window, so this is how we "see" a robot. It saves one PNG per
view to $XLE_LOGS/render/ and prints where each body ended up, which tells us the
robot's forward direction and which arm is physically on which side.

  sbatch --output=$HOME/xle_render_%j.out slurm/sim_job.sbatch \
      /isaac-sim/python.sh tools/render_usd.py <file.usd> --enable_cameras
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("usd")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
simulation_app = AppLauncher(args).app

import imageio.v2 as imageio
import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sensors import Camera, CameraCfg

# Camera positions in metres, all looking at the middle of the robot.
VIEWS = {"plus_x": (1.5, 0, 0.8), "minus_x": (-1.5, 0, 0.8), "plus_y": (0, 1.5, 0.8), "top": (0.01, 0, 2.2)}
TARGET = (0.0, 0.0, 0.4)


def main():
    out_dir = os.path.join(os.environ["XLE_LOGS"], "render")
    os.makedirs(out_dir, exist_ok=True)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 60, device=args.device))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))
    light.func("/World/light", light)

    robot = Articulation(
        ArticulationCfg(
            prim_path="/World/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=args.usd,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True),
            ),
            # Stiff enough to hold the default pose while we take the pictures.
            actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=100.0, damping=10.0)},
        )
    )
    cameras = {}
    for name in VIEWS:
        cameras[name] = Camera(
            CameraCfg(
                prim_path=f"/World/view_{name}",
                height=480,
                width=640,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 20.0)),
            )
        )

    sim.reset()
    for name, eye in VIEWS.items():
        cameras[name].set_world_poses_from_view(
            torch.tensor([eye], device=sim.device, dtype=torch.float32),
            torch.tensor([TARGET], device=sim.device, dtype=torch.float32),
        )

    dt = sim.get_physics_dt()
    for _ in range(90):
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)
        for camera in cameras.values():
            camera.update(dt)

    print("[render] joints:", robot.joint_names)
    print("[render] body positions after settling (x, y, z in metres):")
    for name, pos in zip(robot.body_names, robot.data.body_pos_w[0].cpu().numpy()):
        print(f"[render]   {name:26} {np.round(pos, 3)}")

    for name, camera in cameras.items():
        image = camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        imageio.imwrite(os.path.join(out_dir, f"{name}.png"), image)
        print(f"[render] {name}: mean={image.mean():.1f}")


if __name__ == "__main__":
    main()
    # Kit can hang on shutdown with hand-built scenes, and nothing here needs a clean exit.
    os._exit(0)
