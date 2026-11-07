"""check_dataset.py - load a LeRobot dataset with lerobot itself and describe it.

Prints the feature names and shapes, the episode count and lengths, and the min/max of
every number feature. It also writes one MP4 per episode with the three cameras side by
side, so you can watch what was recorded.

No simulator needed, but it uses the container's Python, which has lerobot installed:

  slurm/in_container.sh /isaac-sim/python.sh tools/check_dataset.py
"""

import argparse
import os

import imageio.v2 as imageio
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo_id", default="local/xle_sandbox_v0")
parser.add_argument("--out_dir", default=os.path.join(os.environ["XLE_LOGS"], "dataset_check"))
args = parser.parse_args()


def main():
    root = os.path.join(os.environ["HF_LEROBOT_HOME"], args.repo_id)
    # pyav decodes the videos with the libraries bundled in the `av` wheel.
    dataset = LeRobotDataset(args.repo_id, root=root, video_backend="pyav")
    meta = dataset.meta
    print(f"[check] {root}")
    print(f"[check] fps {meta.fps}, robot_type {meta.robot_type}, {meta.total_episodes} episodes, {meta.total_frames} frames")

    print("[check] features:")
    for name, feature in meta.features.items():
        names = feature.get("names")
        shown = names if isinstance(names, list) and len(names) <= 17 else ""
        print(f"[check]   {name:32} {feature['dtype']:8} {tuple(feature['shape'])} {shown}")

    episode_index = np.array(dataset.hf_dataset["episode_index"])
    episodes = sorted(set(episode_index.tolist()))
    print("[check] episode lengths:", {int(e): int((episode_index == e).sum()) for e in episodes})

    print("[check] min / max per value:")
    for name in ("action", "observation.state"):
        stats = meta.stats[name]
        for key, low, high in zip(meta.features[name]["names"], np.ravel(stats["min"]), np.ravel(stats["max"])):
            print(f"[check]   {name:18} {key:28} {low:9.3f} {high:9.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    cameras = [k for k in meta.features if k.startswith("observation.images.")]
    for e in episodes:
        frames = []
        for i in np.flatnonzero(episode_index == e):
            item = dataset[int(i)]
            images = [(item[c].permute(1, 2, 0).numpy() * 255).astype(np.uint8) for c in cameras]
            frames.append(np.concatenate(images, axis=1))
        path = os.path.join(args.out_dir, f"episode_{e:03d}.mp4")
        imageio.mimsave(path, frames, fps=meta.fps)
        print(f"[check] wrote {path} ({len(frames)} frames, cameras {[c.split('.')[-1] for c in cameras]})")


if __name__ == "__main__":
    main()
