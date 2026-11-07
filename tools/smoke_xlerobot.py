"""smoke_xlerobot.py - check that XLeRobot drives, moves and sees.

Runs a fixed routine on a plain floor: drive 1 m forward, 1 m left, turn 90 degrees, move
each arm through 3 poses, open and close each gripper, then pan and tilt the head. Saves
PNGs and MP4s from the 3 robot cameras and a third-person camera to $XLE_LOGS/smoke_xlerobot/.

Pass: the robot stays upright, no joint blows up, the base error is below 5 cm and 5 degrees.

  sbatch --output=$HOME/xle_xsmoke_%j.out slurm/sim_job.sbatch \
      /isaac-sim/python.sh tools/smoke_xlerobot.py --enable_cameras
"""

import argparse
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
simulation_app = AppLauncher(args).app

import imageio.v2 as imageio
import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from xle_leisaac.robots.xlerobot import (
    BASE_JOINTS,
    HEAD_CAMERA_CFG,
    HEAD_JOINTS,
    LEFT_ARM_JOINTS,
    LEFT_WRIST_CAMERA_CFG,
    RIGHT_ARM_JOINTS,
    RIGHT_WRIST_CAMERA_CFG,
    XLEROBOT_CFG,
    XLEROBOT_JOINT_NAMES,
    body_to_joint_velocity,
)

DT = 1 / 60
CAMERAS = ("head", "left_wrist", "right_wrist", "third_person")


@configclass
class SmokeSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2500.0))
    # A red box in front of the robot gives the cameras something to look at.
    marker = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/marker",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.3, 0.8), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1))
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.5, 1.0, 0.4)),
    )
    robot = XLEROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    head = HEAD_CAMERA_CFG
    left_wrist = LEFT_WRIST_CAMERA_CFG
    right_wrist = RIGHT_WRIST_CAMERA_CFG
    third_person = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/third_person",
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 30.0)),
        width=640,
        height=480,
    )


class Routine:
    """Steps the sim while holding position targets and recording every second frame."""

    def __init__(self, sim, scene):
        self.sim, self.scene = sim, scene
        self.robot = scene["robot"]
        self.frames = {name: [] for name in CAMERAS}
        self.count = 0
        self.max_speed = 0.0
        self.pos_target = self.robot.data.default_joint_pos.clone()
        self.base_ids = self.robot.find_joints(BASE_JOINTS, preserve_order=True)[0]

    def yaw(self):
        return float(self.robot.data.joint_pos[0, self.base_ids[2]])

    def run(self, seconds, forward=0.0, left=0.0, turn=0.0):
        for _ in range(round(seconds / DT)):
            velocity = torch.zeros_like(self.robot.data.joint_vel)
            velocity[0, self.base_ids] = torch.tensor(
                body_to_joint_velocity(forward, left, turn, self.yaw()), device=velocity.device
            )
            self.robot.set_joint_position_target(self.pos_target)
            self.robot.set_joint_velocity_target(velocity)
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(DT)
            self.max_speed = max(self.max_speed, float(self.robot.data.joint_vel.abs().max()))
            self.count += 1
            if self.count % 2 == 0:
                for name in CAMERAS:
                    image = self.scene[name].data.output["rgb"][0, ..., :3]
                    self.frames[name].append(image.cpu().numpy().astype(np.uint8))

    def set_joints(self, names, values):
        ids = self.robot.find_joints(names, preserve_order=True)[0]
        self.pos_target[0, ids] = torch.tensor(values, device=self.pos_target.device)
        return ids

    def snapshot(self, label, out_dir):
        for name in CAMERAS:
            imageio.imwrite(os.path.join(out_dir, f"{label}_{name}.png"), self.frames[name][-1])


def main():
    out_dir = os.path.join(os.environ["XLE_LOGS"], "smoke_xlerobot")
    os.makedirs(out_dir, exist_ok=True)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args.device))
    scene = InteractiveScene(SmokeSceneCfg(num_envs=1, env_spacing=8.0))
    sim.reset()
    scene["third_person"].set_world_poses_from_view(
        torch.tensor([[2.6, -3.2, 2.0]], device=sim.device), torch.tensor([[0.8, 0.6, 0.5]], device=sim.device)
    )
    robot = scene["robot"]
    print("[xsmoke] sim joints:", robot.joint_names)
    print("[xsmoke] action order:", XLEROBOT_JOINT_NAMES)

    routine = Routine(sim, scene)
    routine.run(1.0)
    routine.snapshot("0_start", out_dir)
    base_start = robot.data.joint_pos[0, routine.base_ids].clone()

    # Drive: 0.25 m/s for 4 s is 1 m. Each leg ends with a short pause.
    routine.run(4.0, forward=0.25)
    routine.run(0.5)
    routine.run(4.0, left=0.25)
    routine.run(0.5)
    routine.run(3.0, turn=math.pi / 6)
    routine.run(0.5)
    routine.snapshot("1_after_drive", out_dir)

    moved = robot.data.joint_pos[0, routine.base_ids] - base_start
    error_m = math.hypot(float(moved[0]) - 1.0, float(moved[1]) - 1.0)
    error_deg = abs(math.degrees(float(moved[2])) - 90.0)
    print(f"[xsmoke] base moved x={float(moved[0]):.3f} y={float(moved[1]):.3f} yaw={math.degrees(float(moved[2])):.2f}")
    print(f"[xsmoke] base error {error_m * 100:.2f} cm, {error_deg:.2f} deg")

    # Arms: three poses for each arm (shoulder pan, lift, elbow, wrist pitch, wrist roll).
    poses = [(0.0, 1.57, 1.57, 0.0, 1.57), (0.5, 1.0, 1.2, 0.5, 0.0), (-0.5, 0.4, 0.6, -0.4, 1.57)]
    worst = 0.0
    for arm_name, joints in (("left", LEFT_ARM_JOINTS), ("right", RIGHT_ARM_JOINTS)):
        for i, pose in enumerate(poses):
            ids = routine.set_joints(joints[:5], list(pose))
            routine.run(2.5)
            error = (robot.data.joint_pos[0, ids] - routine.pos_target[0, ids]).abs().max()
            worst = max(worst, float(error))
            print(f"[xsmoke] {arm_name} arm pose {i}: max joint error {math.degrees(float(error)):.2f} deg")
        routine.snapshot(f"2_{arm_name}_arm", out_dir)
        for value in (1.2, 0.0):
            ids = routine.set_joints(joints[5:], [value])
            routine.run(1.5)
            print(f"[xsmoke] {arm_name} gripper target {value}: at {float(robot.data.joint_pos[0, ids[0]]):.3f}")

    # Head: pan left and right, then tilt down and up.
    for pan, tilt in ((0.7, 0.0), (-0.7, 0.0), (0.0, 0.6), (0.0, -0.4), (0.0, 0.0)):
        ids = routine.set_joints(HEAD_JOINTS, [pan, tilt])
        routine.run(1.5)
        error = (robot.data.joint_pos[0, ids] - routine.pos_target[0, ids]).abs().max()
        print(f"[xsmoke] head pan={pan} tilt={tilt}: max error {math.degrees(float(error)):.2f} deg")
    routine.snapshot("3_end", out_dir)

    # Upright: the body's z axis should still point straight up.
    w, x, y, z = robot.data.root_quat_w[0].tolist()
    tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))
    finite = bool(torch.isfinite(robot.data.joint_pos).all())
    print(f"[xsmoke] root tilt {tilt_deg:.2f} deg, finite={finite}, max joint speed {routine.max_speed:.2f}")
    print(f"[xsmoke] worst arm error {math.degrees(worst):.2f} deg")

    for name, frames in routine.frames.items():
        imageio.mimsave(os.path.join(out_dir, f"{name}.mp4"), frames, fps=30)
        print(f"[xsmoke] {name}: {len(frames)} frames, last mean={frames[-1].mean():.1f}")
    passed = error_m < 0.05 and error_deg < 5 and finite and tilt_deg < 5
    print(f"[xsmoke] RESULT {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
    # Kit hung on shutdown with hand-built scenes, and nothing here needs a clean exit.
    os._exit(0)
