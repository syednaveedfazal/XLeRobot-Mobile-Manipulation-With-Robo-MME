"""replay_episode.py - replay one recorded episode and check the sim repeats it.

It puts the scene back into the episode's recorded starting state (robot and object
poses), then sends the recorded actions one by one. After every step it compares the
robot's joints with the joints recorded at the same step. If recording and replay agree,
the dataset really describes what the robot did.

  sbatch --output=$HOME/xle_replay_%j.out slurm/sim_job.sbatch \
      /isaac-sim/python.sh tools/replay_episode.py --enable_cameras --episode 0
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset_file", default=os.path.join(os.environ["XLE_DATA"], "hdf5", "xle_sandbox_v0.hdf5"))
parser.add_argument("--episode", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab_tasks.utils import parse_env_cfg

import xle_leisaac  # noqa: F401  (registers XLeRobot-Sandbox-v0)

TASK = "XLeRobot-Sandbox-v0"


def main():
    handler = HDF5DatasetFileHandler()
    handler.open(args.dataset_file)
    names = list(handler.get_episode_names())
    episode = handler.load_episode(names[args.episode], device=args.device)
    print(f"[replay] {args.dataset_file}: {len(names)} episodes, replaying {names[args.episode]}")

    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=1)
    env_cfg.terminations.time_out = None  # the replay decides when to stop
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    env.reset()
    env.reset_to(episode.data["initial_state"], torch.tensor([0], device=env.device), is_relative=True)

    actions = episode.data["actions"]
    recorded = episode.data["states"]["articulation"]["robot"]["joint_position"]  # after each step
    robot = env.scene["robot"]
    worst, worst_step = 0.0, 0
    for t in range(actions.shape[0]):
        env.step(actions[t].unsqueeze(0))
        error = float((robot.data.joint_pos[0] - recorded[t]).abs().max())
        if error > worst:
            worst, worst_step = error, t

    cube_now = env.scene["cube_red"].data.root_pos_w[0] - env.scene.env_origins[0]
    cube_rec = episode.data["states"]["rigid_object"]["cube_red"]["root_pose"][-1][:3]
    print(f"[replay] {actions.shape[0]} steps; worst joint difference {worst:.5f} (rad or m) at step {worst_step}")
    print(f"[replay] red cube end position: replay {cube_now.tolist()} vs recorded {cube_rec.tolist()}")
    print(f"[replay] RESULT {'PASS' if worst < 0.01 else 'FAIL'} (threshold 0.01)")
    handler.close()
    env.close()


if __name__ == "__main__":
    # Fail fast: if anything raises, print it and exit now. Otherwise Kit can hang on
    # shutdown and the job sits idle until its time limit.
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    os._exit(0)
