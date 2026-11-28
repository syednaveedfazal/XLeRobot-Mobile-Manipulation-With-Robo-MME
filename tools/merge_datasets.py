"""merge_datasets.py - merge per-chunk LeRobot datasets into one (LeRobot's aggregate_datasets).

Converting one big recording is slow (one process, ~17 frames/s), so each recorded chunk is
converted on its own, in parallel, into <repo>_c<N>, and this script merges them, in chunk
order, into <repo>. It also writes meta/xle_episodes.json: the hidden station and the random
timing of every merged episode, in dataset episode order, from the chunks' side files.

  slurm/in_container.sh /isaac-sim/python.sh tools/merge_datasets.py --repo_id local/xle_put_it_back_v1 \
      --hdf5_dir $XLE_DATA/hdf5 --tag put_it_back_v1
"""

import argparse
import glob
import json
import os
import re
import shutil
import time
from pathlib import Path

from lerobot.datasets import lerobot_dataset
from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata


def _local_only(repo_id, version):
    """Our datasets exist only on Marvin. LeRobot asks the Hugging Face Hub for a dataset's
    version when its local metadata does not load; fail here instead of contacting the Hub."""
    raise FileNotFoundError(f"{repo_id}: local metadata missing or incomplete")


lerobot_dataset.get_safe_version = _local_only


def fix_video_timestamps(root):
    """Repair LeRobot 0.4.2's aggregate_datasets: when it starts a new video file, it keeps adding
    the earlier files' durations to the episodes' from/to timestamps instead of restarting at 0, so
    frames are looked up far past the end of the file ("query timestamps violate the tolerance").
    Recompute them from the episode lengths, in order within each file, and check that every
    file has exactly as many frames as its episodes."""
    import av
    import pandas as pd

    root = Path(root)
    fps = json.loads((root / "meta" / "info.json").read_text())["fps"]
    files = sorted(glob.glob(str(root / "meta" / "episodes" / "*" / "*.parquet")))
    eps = pd.concat([pd.read_parquet(f).assign(_file=f) for f in files], ignore_index=True)
    keys = sorted({c.split("/")[1] for c in eps.columns if c.startswith("videos/") and c.endswith("/from_timestamp")})
    for key in keys:
        ci, fi = f"videos/{key}/chunk_index", f"videos/{key}/file_index"
        for (c, f), group in eps.sort_values("episode_index").groupby([ci, fi], sort=False):
            starts = group["length"].cumsum().shift(fill_value=0) / fps
            eps.loc[group.index, f"videos/{key}/from_timestamp"] = starts.values
            eps.loc[group.index, f"videos/{key}/to_timestamp"] = (starts + group["length"] / fps).values
            video = root / "videos" / key / f"chunk-{int(c):03d}" / f"file-{int(f):03d}.mp4"
            with av.open(str(video)) as container:
                frames = sum(1 for p in container.demux(container.streams.video[0]) if p.pts is not None)
            if frames != int(group["length"].sum()):
                raise RuntimeError(f"{video}: {frames} frames but its episodes have {int(group['length'].sum())}")
    import pyarrow as pa
    import pyarrow.parquet as pq

    for f in files:  # replace only the timestamp columns, keep every other column's type as written
        table, part = pq.read_table(f), eps[eps["_file"] == f]
        for key in keys:
            for col in (f"videos/{key}/from_timestamp", f"videos/{key}/to_timestamp"):
                i = table.schema.get_field_index(col)
                table = table.set_column(i, table.schema.field(i), pa.array(part[col].to_numpy(), type=table.schema.field(i).type))
        pq.write_table(table, f)
    print(f"[merge] video timestamps recomputed for {len(eps)} episodes x {len(keys)} cameras", flush=True)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--repo_id", required=True, help="merged dataset; parts are <repo_id>_c<N>")
parser.add_argument("--hdf5_dir", required=True, help="where the chunks' <tag>_c<N>_episodes.json side files are")
parser.add_argument("--tag", required=True, help="HDF5 chunk name prefix, e.g. put_it_back_v1")
args = parser.parse_args()

home = Path(os.environ["HF_LEROBOT_HOME"])
parts = sorted(glob.glob(str(home / f"{args.repo_id}_c*")), key=lambda p: int(re.search(r"_c(\d+)$", p).group(1)))
good = []
for p in parts:  # keep only chunks whose metadata loads completely (a failed conversion leaves a stub)
    try:
        meta = LeRobotDatasetMetadata(str(Path(p).relative_to(home)), root=p)
        if meta.total_episodes > 0:
            good.append(p)
            continue
    except Exception as e:  # noqa: BLE001
        print(f"[merge] skipping {Path(p).name}: {type(e).__name__}: {e}", flush=True)
        continue
    print(f"[merge] skipping {Path(p).name}: no episodes", flush=True)
parts = good
if not parts:
    raise SystemExit("no converted chunks found")
target = home / args.repo_id
if target.exists():  # keep the previous merge
    old = home / "_archive" / f"{args.repo_id.replace('/', '_')}_{time.strftime('%Y%m%d_%H%M%S')}"
    old.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(target), str(old))
    print(f"[merge] moved the previous {args.repo_id} to {old}", flush=True)

repo_ids = [str(Path(p).relative_to(home)) for p in parts]
print(f"[merge] {len(parts)} chunks: {', '.join(Path(p).name for p in parts)}", flush=True)
aggregate_datasets(repo_ids=repo_ids, aggr_repo_id=args.repo_id, roots=[Path(p) for p in parts], aggr_root=target)
fix_video_timestamps(target)

# Side file: hidden station and random timing per merged episode (successful ones, in order).
rows = []
for p in parts:
    n = int(re.search(r"_c(\d+)$", p).group(1))
    side = Path(args.hdf5_dir) / f"{args.tag}_c{n}_episodes.json"
    kept = json.loads((Path(p) / "meta" / "info.json").read_text())["total_episodes"]
    eps = [dict(chunk=n, **e) for e in json.loads(side.read_text())["episodes"] if e["success"]] if side.exists() else []
    # LeIsaac's converter writes a chunk's demos in text order of their names (demo_0, demo_1, demo_10, ...).
    eps.sort(key=lambda e: str(e["episode"]))
    if len(eps) != kept:
        print(f"[merge] WARNING chunk {n}: {kept} episodes converted but {len(eps)} successful in {side.name}", flush=True)
    rows += eps[:kept] + [dict(chunk=n, station=None)] * max(0, kept - len(eps))
for i, r in enumerate(rows):
    r["dataset_episode"] = i
(target / "meta" / "xle_episodes.json").write_text(json.dumps(rows, indent=1))
info = json.loads((target / "meta" / "info.json").read_text())
stations = [r.get("station") for r in rows]
print(f"[merge] {args.repo_id}: {info['total_episodes']} episodes, {info['total_frames']} frames; "
      f"stations {', '.join(f'{k}: {stations.count(k)}' for k in (0, 1, 2))}", flush=True)


def timing_check(rows, folds=5, limit=0.45, seed=0):
    """B12 gate: can the time of the turn-back (the memory decision point) alone tell the station?
    5-fold cross-validated 1-D Gaussian classifier per station, as the data-auditor ran it on D2
    (65 % there; chance is 33 %). Above `limit`, or with more than 5 % late episodes, the dataset
    still has a timing shortcut and must not be trained on."""
    import numpy as np

    r = [x for x in rows if x.get("turn_back_at") is not None and x.get("station") is not None]
    if not r:
        print("[merge] timing check skipped: no turn_back_at in the side files (recorded before the B12 fix)", flush=True)
        return True
    t, y = np.array([x["turn_back_at"] for x in r]), np.array([x["station"] for x in r])
    order = np.random.default_rng(seed).permutation(len(t))
    hits = 0
    for k in range(folds):
        test = order[k::folds]
        train = np.setdiff1d(order, test)
        stats = [(t[train][y[train] == c].mean(), t[train][y[train] == c].std() + 1e-6) for c in (0, 1, 2)]
        logp = np.stack([-np.log(sd) - 0.5 * ((t[test] - m) / sd) ** 2 for m, sd in stats], axis=1)
        hits += int((logp.argmax(1) == y[test]).sum())
    acc, late = hits / len(t), sum(bool(x.get("late")) for x in r)
    per = ", ".join(f"{c}: {t[y == c].mean():.1f} s ({t[y == c].min():.1f}-{t[y == c].max():.1f})" for c in (0, 1, 2))
    print(f"[merge] turn-back time per station: {per}; late episodes {late}/{len(t)}", flush=True)
    print(f"[merge] station from the turn-back time alone: {acc:.0%} (chance 33 %, limit {limit:.0%})", flush=True)
    if acc > limit or late > 0.05 * len(t):
        print("[merge] STOP: the turn-back time still gives the station away (B12); do not train on this dataset", flush=True)
        return False
    return True


if not timing_check(rows):
    raise SystemExit(1)

