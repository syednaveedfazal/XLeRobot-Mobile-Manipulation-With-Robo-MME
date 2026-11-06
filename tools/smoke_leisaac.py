"""smoke_leisaac.py - prove that Isaac Sim, Isaac Lab and LeIsaac work together headless.

It loads a LeIsaac environment, waves the arm for 300 steps, and saves one PNG per
camera plus a 10 s MP4 to $XLE_LOGS/smoke/. It also prints steps per second and the
brightness of each camera, because a black image is the classic failure on a cluster.

Run inside a GPU job:
  sbatch --output=$HOME/xle_smoke_%j.out slurm/sim_job.sbatch \
      /isaac-sim/python.sh tools/smoke_leisaac.py --enable_cameras
"""

import argparse
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="LeIsaac-SO101-PickOrange-v0")
parser.add_argument("--steps", type=int, default=300)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

# Isaac Sim must start before anything else from isaaclab or leisaac is imported.
simulation_app = AppLauncher(args).app

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from isaaclab_tasks.utils import parse_env_cfg

import leisaac  # noqa: F401  (importing it registers the LeIsaac environments)

CAMERAS = ("front", "wrist")


def main():
    out_dir = os.path.join(os.environ["XLE_LOGS"], "smoke")
    os.makedirs(out_dir, exist_ok=True)

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.use_teleop_device("so101leader")  # actions become plain joint positions
    env_cfg.seed = 0
    env_cfg.terminations.time_out = None  # keep running for all the steps
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None
    env = gym.make(args.task, cfg=env_cfg).unwrapped

    obs, _ = env.reset()
    start_pose = env.scene["robot"].data.joint_pos[:, :6].clone()
    videos = {name: [] for name in CAMERAS}

    start = time.time()
    for step in range(args.steps):
        action = start_pose.clone()
        action[:, 0] += 0.3 * np.sin(step / 30)  # sway the shoulder
        with torch.inference_mode():
            obs, *_ = env.step(action)
        for name in CAMERAS:
            videos[name].append(obs["policy"][name][0].cpu().numpy().astype(np.uint8))
    elapsed = time.time() - start
    print(f"[smoke] {args.steps} steps in {elapsed:.1f} s = {args.steps / elapsed:.1f} steps/s")

    for name, frames in videos.items():
        imageio.imwrite(os.path.join(out_dir, f"{name}.png"), frames[-1])
        imageio.mimsave(os.path.join(out_dir, f"{name}.mp4"), frames, fps=30)
        print(f"[smoke] {name}: shape={frames[-1].shape} mean={frames[-1].mean():.1f} std={frames[-1].std():.1f}")
        if frames[-1].max() == 0:
            print(f"[smoke] FAIL: {name} is completely black")

    print(f"[smoke] saved to {out_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
