"""record_scripted.py - record scripted XLeRobot episodes in the sandbox, headless, to HDF5.

Each episode plays the same routine: drive up to the table, look down at it, reach with the
left arm and open/close its gripper, the same with the right arm, then look around. The
objects are shuffled on every reset (from --seed) and the driving distance varies a little,
so the episodes differ.

Recording uses LeIsaac's StreamingRecorderManager, which writes to the HDF5 file every 100
steps instead of keeping a whole episode of images in memory.

LeIsaac's converter only keeps episodes marked "successful". The sandbox has no success
condition, so this script adds one that fires when the routine has finished: "success"
here just means "the episode was recorded completely".

  sbatch --output=$HOME/xle_record_%j.out slurm/sim_job.sbatch \
      /isaac-sim/python.sh tools/record_scripted.py --enable_cameras --episodes 5
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--dataset_file", default=os.path.join(os.environ["XLE_DATA"], "hdf5", "xle_sandbox_v0.hdf5"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
simulation_app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
from isaaclab.managers import DatasetExportMode
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.enhance.managers import StreamingRecorderManager

import xle_leisaac  # noqa: F401  (registers XLeRobot-Sandbox-v0)
from xle_leisaac.robots.xlerobot import REST_ARM_POSE

TASK = "XLeRobot-Sandbox-v0"
FPS = 30
ROUTINE_DONE = {"flag": False}

# Arm poses for one arm, in radians, in SO-101 joint order (pan, lift, elbow, wrist flex, roll, gripper).
REST = [REST_ARM_POSE[k] for k in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")]
REACH = [0.0, 0.6, 0.3, 0.9, 0.0, -0.17]
REACH_OPEN = REACH[:5] + [1.2]


def routine_finished(env):
    return torch.full((env.num_envs,), ROUTINE_DONE["flag"], dtype=torch.bool, device=env.device)


def make_routine(drive_m):
    """List of keyframes: (seconds to get there, 14 joint targets, base command).

    Joint targets are left arm (6), right arm (6), head pan and tilt. The base command is
    [forward m/s, left m/s, turn rad/s] and is held for the whole keyframe.
    """
    steps = []

    def keyframe(seconds, left, right, head, base=(0.0, 0.0, 0.0)):
        steps.append((seconds, list(left) + list(right) + list(head), base))

    look_ahead, look_down = (0.0, 0.0), (0.0, 0.45)
    keyframe(1.0, REST, REST, look_ahead)                                 # settle
    keyframe(3.0, REST, REST, look_ahead, base=(drive_m / 3.0, 0, 0))     # drive to the table
    keyframe(0.5, REST, REST, look_ahead)                                 # stop
    keyframe(1.0, REST, REST, look_down)                                  # look at the table
    for side in ("left", "right"):
        for seconds, pose in ((2.0, REACH), (1.0, REACH_OPEN), (1.0, REACH), (2.0, REST)):
            left, right = (pose, REST) if side == "left" else (REST, pose)
            keyframe(seconds, left, right, look_down)                     # reach, open, close, return
    for pan in (0.6, -0.6, 0.0):
        keyframe(1.5, REST, REST, (pan, 0.45))                            # look around
    return steps


def main():
    os.makedirs(os.path.dirname(args.dataset_file), exist_ok=True)
    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=1)
    env_cfg.seed = args.seed
    env_cfg.terminations.success = DoneTerm(func=routine_finished)
    env_cfg.recorders.dataset_export_dir_path = os.path.dirname(args.dataset_file)
    env_cfg.recorders.dataset_filename = os.path.splitext(os.path.basename(args.dataset_file))[0]
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    env = gym.make(TASK, cfg=env_cfg).unwrapped
    del env.recorder_manager  # swap in LeIsaac's streaming recorder, as its teleop script does
    env.recorder_manager = StreamingRecorderManager(env_cfg.recorders, env)
    env.recorder_manager.flush_steps = 100
    env.recorder_manager.compression = "lzf"

    rng = np.random.default_rng(args.seed)
    env.reset()
    robot = env.scene["robot"]
    for episode in range(args.episodes):
        routine = make_routine(drive_m=float(rng.uniform(0.72, 0.76)))
        start = torch.tensor(routine[0][1], device=env.device)
        frames = 0
        for seconds, target, base in routine:
            target = torch.tensor(target, device=env.device)
            n = max(1, round(seconds * FPS))
            for i in range(n):
                joints = start + (target - start) * (i + 1) / n  # straight-line move between keyframes
                action = torch.cat([joints, torch.tensor(base, device=env.device)]).unsqueeze(0)
                env.step(action)
                frames += 1
            start = target
        ROUTINE_DONE["flag"] = True
        env.step(action)  # the success term fires here; the env resets and the recorder exports
        ROUTINE_DONE["flag"] = False
        pose = robot.data.joint_pos[0, :3].tolist()
        print(f"[record] episode {episode}: {frames + 1} frames, "
              f"exported so far: {env.recorder_manager.exported_successful_episode_count}, base after reset {pose}")

    if hasattr(env.recorder_manager, "finalize"):
        env.recorder_manager.finalize()
    print(f"[record] wrote {args.dataset_file}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
