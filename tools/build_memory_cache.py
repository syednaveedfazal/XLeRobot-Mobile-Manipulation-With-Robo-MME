"""build_memory_cache.py - pre-compute memory features for training pi05_mem.

Runs every frame of one camera through the policy's frozen SigLIP + projector, pools
it to 4x4 tokens and stores the result (float16) row by row, so training can look up
any frame's history without decoding video. ~64 KB per frame for pi0.5 (16 x 2048).
Rebuild when the dataset or the starting weights change.

  XLE_WITH_PI=1 sbatch --output=$HOME/xle_cache_%j.out slurm/sim_job.sbatch \
      /isaac-sim/python.sh tools/build_memory_cache.py --repo_id local/xle_sandbox_v0

It prints the --policy.memory_feature_dir to pass to tools/train_policy.py.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import xle_policies  # noqa: F401
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from xle_policies.pi05_mem import PI05MemConfig, PI05MemPolicy
from xle_policies.pi05_mem.memory import image_feature_dim

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--repo_id", required=True)
parser.add_argument("--root", default=None, help="dataset folder (default $HF_LEROBOT_HOME/<repo_id>)")
parser.add_argument("--camera", default="observation.images.head")
parser.add_argument("--pretrained_path", default="lerobot/pi05_base",
                    help="weights whose vision encoder is used; must match training's --policy.pretrained_path")
parser.add_argument("--paligemma_variant", default="gemma_2b")
parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
parser.add_argument("--tokens_per_frame", type=int, default=16)
parser.add_argument("--pool", default="mean", choices=["mean", "max"])
parser.add_argument("--out", default=None, help="default <dataset root>/memory_features/<camera>")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--pad_black", type=int, default=1, help="1: pad the 224 px letterbox with black (as new training runs do)")
parser.add_argument("--num_workers", type=int, default=8)
args = parser.parse_args()

# PyAV: torchcodec (LeRobot's default) needs system FFmpeg libraries the container does not have.
ds = LeRobotDataset(args.repo_id, root=args.root, video_backend="pyav")
if args.camera not in ds.meta.camera_keys:
    raise SystemExit(f"{args.camera} not in dataset cameras {ds.meta.camera_keys}")
n = ds.meta.total_frames
out = Path(args.out or Path(ds.root) / "memory_features" / args.camera.split(".")[-1])
out.mkdir(parents=True, exist_ok=True)

# Every frame's episode start, from the episode table (rows are contiguous per episode).
episode_start = np.full(n, -1, dtype=np.int64)
for a, b in zip(ds.meta.episodes["dataset_from_index"], ds.meta.episodes["dataset_to_index"], strict=True):
    episode_start[a:b] = a
if (episode_start < 0).any():
    raise SystemExit("episode table does not cover every dataset row")

features = dataset_to_policy_features(ds.meta.features)
cfg = PI05MemConfig(
    memory="framesamp_modul",
    memory_camera=args.camera,
    memory_tokens_per_frame=args.tokens_per_frame,
    memory_pool=args.pool,
    paligemma_variant=args.paligemma_variant,
    dtype=args.dtype,
    device="cuda" if torch.cuda.is_available() else "cpu",
    pretrained_path=args.pretrained_path,
    pad_black=bool(args.pad_black),
)
cfg.output_features = {k: f for k, f in features.items() if f.type is FeatureType.ACTION}
cfg.input_features = {k: f for k, f in features.items() if k not in cfg.output_features}
if args.pretrained_path:
    policy = PI05MemPolicy.from_pretrained(args.pretrained_path, config=cfg)
else:
    print("[cache] WARNING: no --pretrained_path, using random weights (tests only)")
    policy = PI05MemPolicy(cfg)
policy.to(cfg.device).eval()

dim = image_feature_dim(policy.model)
tmp = out / "features.tmp.npy"
store = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(n, args.tokens_per_frame, dim))
seen = np.zeros(n, dtype=bool)

loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
t0 = time.time()
for i, batch in enumerate(loader):
    rows = batch["index"].numpy()
    feats = policy.encode_memory_frames(batch[args.camera]).cpu()
    if not torch.isfinite(feats).all():
        raise SystemExit(f"non-finite features at rows {rows[0]}..{rows[-1]} (float16 overflow?)")
    store[rows] = feats.numpy()
    seen[rows] = True
    if i % 50 == 0:
        print(f"[cache] {seen.sum()}/{n} frames, {time.time() - t0:.0f}s", flush=True)
store.flush()
del store
if not seen.all():
    raise SystemExit(f"{(~seen).sum()} dataset rows were not visited")

os.replace(tmp, out / "features.npy")
np.save(out / "episode_start.npy", episode_start)
meta = {
    "repo_id": args.repo_id,
    "camera": args.camera,
    "num_frames": int(n),
    "num_episodes": int(ds.meta.total_episodes),
    "fps": ds.meta.fps,
    "tokens_per_frame": args.tokens_per_frame,
    "pool": args.pool,
    "dim": int(dim),
    "source_weights": args.pretrained_path,
    "dtype": args.dtype,
    "image_resolution": list(cfg.image_resolution),
    "pad_black": bool(args.pad_black),
    "built": time.strftime("%Y-%m-%d %H:%M:%S"),
}
(out / "meta.json").write_text(json.dumps(meta, indent=2))
print(f"[cache] done: {n} frames in {time.time() - t0:.0f}s")
print(f"[cache] train with: --policy.memory_feature_dir={out}")
