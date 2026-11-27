"""run_put_it_back_expert.py - run the scripted PutItBack expert, report how it does, record demos.

For every episode it prints the station, the outcome (success / wrong_station / dropped /
time_out), the time taken and the last plan step reached, then a success rate. --videos
saves an overview / head / right-wrist MP4 per episode; --record also writes the episodes
to HDF5 with LeIsaac's streaming recorder (only successful ones are kept by the converter).

  sbatch --output=$HOME/xle_pib_%j.out slurm/sim_job.sbatch \
      /isaac-sim/python.sh tools/run_put_it_back_expert.py --episodes 3 --videos
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--videos", action="store_true")
parser.add_argument("--record", default=None, help="HDF5 file to record the episodes to")
parser.add_argument("--noise", action="store_true", help="recovery demos: execute slightly noisy actions, record clean ones")
parser.add_argument("--random_stations", action="store_true", help="random station per episode even when recording")
parser.add_argument("--balanced", action="store_true", help="cycle the stations 0, 1, 2, ... also when not recording")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.managers import DatasetExportMode  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import xle_leisaac  # noqa: F401, E402
from xle_leisaac.experts.action_noise import ActionNoise  # noqa: E402
from xle_leisaac.experts.put_it_back_expert import PutItBackExpert  # noqa: E402
from xle_leisaac.tasks.put_it_back_env_cfg import LAYOUT  # noqa: E402

TASK = "XLeRobot-PutItBack-v0"
OUTCOMES = ("success", "wrong_station", "dropped", "time_out")


def main():
    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=1)
    env_cfg.seed = args.seed
    env_cfg.balanced_stations = (bool(args.record) and not args.random_stations) or args.balanced  # demos split evenly over stations
    if args.record:
        os.makedirs(os.path.dirname(os.path.abspath(args.record)), exist_ok=True)
        env_cfg.recorders.dataset_export_dir_path = os.path.dirname(os.path.abspath(args.record))
        env_cfg.recorders.dataset_filename = os.path.splitext(os.path.basename(args.record))[0]
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
    else:
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_NONE  # no files when only testing
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    if args.record:
        from leisaac.enhance.managers import StreamingRecorderManager

        del env.recorder_manager
        env.recorder_manager = StreamingRecorderManager(env_cfg.recorders, env)
        env.recorder_manager.flush_steps = 100
        env.recorder_manager.compression = "lzf"

    out_dir = os.path.join(os.environ["XLE_LOGS"], "put_it_back_expert")
    os.makedirs(out_dir, exist_ok=True)
    env.reset()
    env.scene["overview"].set_world_poses_from_view(
        torch.tensor([LAYOUT["OVERVIEW_EYE"]], device=env.device) + env.scene.env_origins[:1],
        torch.tensor([LAYOUT["OVERVIEW_TARGET"]], device=env.device) + env.scene.env_origins[:1],
    )
    expert = PutItBackExpert(env, seed=args.seed)
    noise = ActionNoise(env) if args.noise else None
    results, stations, seen, episodes = [], [], [], []
    for episode in range(args.episodes):
        expert.reset()
        if noise:
            noise.reset()
        frames = {"overview": [], "head": [], "right_wrist": []}
        steps, outcome, last_step = 0, None, expert.step_name
        while outcome is None:
            last_step = expert.step_name
            action = expert.act()
            if noise:
                noise.base_enabled = expert.driving and not expert.precise  # the base controller corrects it
                noise.arm_enabled = not expert.precise  # no arm noise while grasping or setting down
            _, _, terminated, truncated, _ = env.step(action)
            steps += 1
            if args.videos and steps % 2 == 0:
                for name in frames:
                    image = env.scene[name].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
                    frames[name].append(image)
            if bool(terminated[0]) or bool(truncated[0]):
                log = env.extras.get("log", {})
                outcome = next((o for o in OUTCOMES if log.get(f"Episode_Termination/{o}", 0.0) > 0.5), "unknown")
        seconds = steps * env.step_dt
        results.append(outcome)
        episodes.append(dict(episode=episode, outcome=outcome, success=outcome == "success", noise=bool(noise),
                             sim_seconds=round(seconds, 1), **expert.episode_info()))
        stations.append(expert.station)
        seen.append(expert.seen)
        w, h = expert.seen_px
        print(f"[expert] episode {episode}: station {expert.station}  {outcome:13} {seconds:5.1f} s  "
              f"cube seen {'at %.0fx%.0f px after %.1f s' % (w, h, expert.seen_at) if expert.seen else 'NEVER clearly'}  "
              f"last step: {last_step}", flush=True)
        ev = expert.events
        print(f"[expert]   at the table {ev.get('hand up and wait', float('nan')):.1f} s, wait until {expert.wait_until:.1f} s, "
              f"turn back {ev.get('turn back', float('nan')):.1f} s{'  LATE' if expert.late else ''}", flush=True)
        if args.videos:
            for name, images in frames.items():
                imageio.mimsave(os.path.join(out_dir, f"ep{episode:03d}_{name}.mp4"), images, fps=15)
    rate = results.count("success") / len(results)
    turn = {k: [e["turn_back_at"] for e in episodes if e["station"] == k and e["turn_back_at"] is not None] for k in sorted(set(stations))}
    print("[expert] turn-back time per station (s, mean / min-max): " + ", ".join(
        f"{k}: {np.mean(v):.1f} / {min(v):.1f}-{max(v):.1f}" for k, v in turn.items() if v)
        + f"; late episodes: {sum(e['late'] for e in episodes)}", flush=True)
    print(f"[expert] stations: " + ", ".join(f"{k}: {stations.count(k)}" for k in sorted(set(stations)))
          + f"; cube clearly seen before driving to it: {sum(seen)}/{len(seen)}"
          + ("; recovery noise ON" if noise else ""), flush=True)
    print(f"[expert] success {results.count('success')}/{len(results)} = {rate:.0%}; outcomes "
          + ", ".join(f"{o}: {results.count(o)}" for o in OUTCOMES + ("unknown",) if results.count(o)), flush=True)
    if args.record:  # side file: the hidden station and the random timing of every episode, in recording order
        import json

        with open(os.path.splitext(args.record)[0] + "_episodes.json", "w") as f:
            json.dump(dict(seed=args.seed, noise=bool(noise), episodes=episodes), f, indent=1)
    if args.record and hasattr(env.recorder_manager, "finalize"):
        env.recorder_manager.finalize()
        print(f"[expert] recorded to {args.record}")
    if args.videos:
        print(f"[expert] videos in {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    os._exit(0)
