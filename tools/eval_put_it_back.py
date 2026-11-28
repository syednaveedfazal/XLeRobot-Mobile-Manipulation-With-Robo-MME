"""eval_put_it_back.py - let a trained pi0.5 policy drive XLeRobot-PutItBack-v0 and score it.

The policy gets exactly what it was trained on: the head and both wrist camera images and the
robot's 17 state values in real-robot units, plus the task sentence. It returns 17 actions in
real units, converted back to sim actions. It is called every control step (30 Hz), as the
memory variant adds every head-camera frame to its memory; it plans 50 actions at a time.

Per episode it records whether the cube was picked up, delivered to the work table, and how
the episode ended (success / wrong_station / dropped / time_out). Use the same --seed for every
policy you compare, so they all see the same stations and cube placements.

  XLE_WITH_PI=1 sbatch --partition=mlgpu_short --time=04:00:00 slurm/sim_job.sbatch \\
      /isaac-sim/python.sh tools/eval_put_it_back.py \\
      --checkpoint $XLE_WORK/train_runs/<run>/checkpoints/last/pretrained_model --episodes 30 --videos 3

Results: $XLE_LOGS/eval_put_it_back/<name>/results.json and epNNN_{overview,head}.mp4.
"""

import argparse
import json
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--checkpoint", required=True, help="a .../checkpoints/<step>/pretrained_model folder")
parser.add_argument("--name", default=None, help="results folder name (default: run name + step)")
parser.add_argument("--episodes", type=int, default=30)
parser.add_argument("--seed", type=int, default=777, help="not used by any training chunk (those use 1000..12000)")
parser.add_argument("--videos", type=int, default=3, help="save videos of the first N episodes")
parser.add_argument("--n_action_steps", type=int, default=None, help="actions executed per plan (default: as trained, 50)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless, args.enable_cameras = True, True
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.managers import DatasetExportMode  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.utils.control_utils import predict_action  # noqa: E402

import xle_leisaac  # noqa: F401, E402
import xle_policies  # noqa: F401, E402  (registers pi05_mem)
from xle_leisaac.robots.xlerobot import lerobot_to_sim_action, sim_to_lerobot_action  # noqa: E402
from xle_leisaac.tasks.put_it_back_env_cfg import LAYOUT  # noqa: E402
from xle_policies.pi05_mem import PI05MemPolicy  # noqa: E402

TASK = "XLeRobot-PutItBack-v0"
OUTCOMES = ("success", "wrong_station", "dropped", "time_out")
CAMERAS = ("head", "left_wrist", "right_wrist")


def main():
    ckpt = os.path.abspath(args.checkpoint)
    run, step = ckpt.split(os.sep)[-4], ckpt.split(os.sep)[-2]
    name = args.name or f"{run}_{step}"
    out_dir = os.path.join(os.environ["XLE_LOGS"], "eval_put_it_back", name)
    os.makedirs(out_dir, exist_ok=True)

    policy = PI05MemPolicy.from_pretrained(ckpt)
    if args.n_action_steps:
        policy.config.n_action_steps = args.n_action_steps
    policy.to("cuda").eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=ckpt)
    print(f"[eval] {name}: memory={policy.config.memory} n_action_steps={policy.config.n_action_steps}", flush=True)

    env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)
    env_cfg.seed = args.seed
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_NONE
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    task = env_cfg.task_description
    obs, _ = env.reset()
    env.scene["overview"].set_world_poses_from_view(
        torch.tensor([LAYOUT["OVERVIEW_EYE"]], device=env.device) + env.scene.env_origins[:1],
        torch.tensor([LAYOUT["OVERVIEW_TARGET"]], device=env.device) + env.scene.env_origins[:1],
    )
    cube = env.scene["cube"]
    results = []

    def save(results):
        n = len(results)
        count = lambda key, val=True: sum(r[key] == val for r in results)  # noqa: E731
        summary = dict(
            name=name, checkpoint=ckpt, memory=policy.config.memory, episodes=n, seed=args.seed,
            picked=count("picked"), delivered=count("delivered"), **{o: count("outcome", o) for o in OUTCOMES},
            right_station_given_delivered=f"{count('outcome', 'success')}/{count('outcome', 'success') + count('outcome', 'wrong_station')}",
        )
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(dict(summary=summary, episodes=results), f, indent=2)
        return summary

    for episode in range(args.episodes):
        policy.reset()
        station = int(env.put_it_back_station[0])
        z0 = float(cube.data.root_pos_w[0, 2])
        picked, delivered, outcome, steps, t0 = False, False, None, 0, time.time()
        frames = {"overview": [], "head": []}
        while outcome is None:
            o = obs["policy"]
            observation = {f"observation.images.{c}": o[c][0].cpu().numpy() for c in CAMERAS}
            observation["observation.state"] = sim_to_lerobot_action(o["state"][:1])[0].astype(np.float32)
            action = predict_action(observation, policy, torch.device("cuda"), pre, post, use_amp=False, task=task)
            sim_action = torch.as_tensor(lerobot_to_sim_action(action.reshape(1, -1)), dtype=torch.float32, device=env.device)
            obs, _, terminated, truncated, _ = env.step(sim_action)
            steps += 1
            picked |= float(cube.data.root_pos_w[0, 2]) - z0 > 0.05
            delivered |= bool(env.put_it_back_delivered[0])
            if episode < args.videos and steps % 2 == 0:
                frames["overview"].append(env.scene["overview"].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8))
                frames["head"].append(obs["policy"]["head"][0].cpu().numpy().astype(np.uint8))
            if bool(terminated[0]) or bool(truncated[0]):
                log = env.extras.get("log", {})
                outcome = next((o for o in OUTCOMES if log.get(f"Episode_Termination/{o}", 0.0) > 0.5), "unknown")
        seconds = steps * env.step_dt
        results.append(dict(episode=episode, station=station, outcome=outcome, picked=picked, delivered=delivered,
                            sim_seconds=round(seconds, 1), wall_seconds=round(time.time() - t0, 1)))
        print(f"[eval] episode {episode}: station {station}  {outcome:13}  picked={picked}  delivered={delivered}  "
              f"{seconds:5.1f} s sim, {time.time() - t0:5.0f} s wall", flush=True)
        for cam, images in frames.items():
            if images:
                imageio.mimsave(os.path.join(out_dir, f"ep{episode:03d}_{cam}.mp4"), images, fps=15)
        summary = save(results)  # after every episode: a job that hits its time limit keeps what it has

    print(f"[eval] summary: {json.dumps(summary)}", flush=True)
    print(f"[eval] results and videos in {out_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    os._exit(0)
