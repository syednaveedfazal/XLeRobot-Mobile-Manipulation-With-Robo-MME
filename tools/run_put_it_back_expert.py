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
from xle_leisaac.experts.put_it_back_expert import PutItBackExpert  # noqa: E402

TASK = "XLeRobot-PutItBack-v0"
OUTCOMES = ("success", "wrong_station", "dropped", "time_out")


def main():
    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=1)
    env_cfg.seed = args.seed
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
        torch.tensor([[0.4, -2.9, 2.3]], device=env.device) + env.scene.env_origins[:1],
        torch.tensor([[0.4, 0.0, 0.6]], device=env.device) + env.scene.env_origins[:1],
    )
    expert = PutItBackExpert(env)
    results = []
    for episode in range(args.episodes):
        expert.reset()
        frames = {"overview": [], "head": [], "right_wrist": []}
        steps, outcome, last_step = 0, None, expert.step_name
        while outcome is None:
            last_step = expert.step_name
            _, _, terminated, truncated, _ = env.step(expert.act())
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
        print(f"[expert] episode {episode}: station {expert.station}  {outcome:13} {seconds:5.1f} s  "
              f"last step: {last_step}", flush=True)
        if args.videos:
            for name, images in frames.items():
                imageio.mimsave(os.path.join(out_dir, f"ep{episode:03d}_{name}.mp4"), images, fps=15)
    rate = results.count("success") / len(results)
    print(f"[expert] success {results.count('success')}/{len(results)} = {rate:.0%}; outcomes "
          + ", ".join(f"{o}: {results.count(o)}" for o in OUTCOMES + ("unknown",) if results.count(o)), flush=True)
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
