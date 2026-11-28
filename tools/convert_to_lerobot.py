"""convert_to_lerobot.py - run LeIsaac's HDF5 -> LeRobot v3 converter on XLeRobot data.

LeIsaac's scripts/convert/isaaclab2lerobotv3.py works for any environment that provides a
few hooks (build_lerobot_frame, default_feature_joint_names, ...). Our sandbox provides
them, so the converter runs unchanged. It only needs our environments registered first,
which is all this wrapper does before handing over to LeIsaac's script.

The dataset lands in $HF_LEROBOT_HOME/<repo_id>, i.e. $XLE_DATA/lerobot/local/xle_sandbox_v0.

  sbatch --output=$HOME/xle_convert_%j.out slurm/sim_job.sbatch \
      /isaac-sim/python.sh tools/convert_to_lerobot.py
"""

import argparse
import os
import runpy
import shutil
import sys

import xle_leisaac  # noqa: F401  (registers XLeRobot-Sandbox-v0; needs no simulator yet)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--hdf5_file", nargs="+", default=[os.path.join(os.environ["XLE_DATA"], "hdf5", "xle_sandbox_v0.hdf5")],
                    help="one or more HDF5 files, all in the same folder (merged into one dataset)")
parser.add_argument("--repo_id", default="local/xle_sandbox_v0")
parser.add_argument("--task", default="XLeRobot-Sandbox-v0", help="gym id the HDF5 was recorded with")
parser.add_argument("--overwrite", action="store_true", help="move an existing dataset with this repo_id to _archive/ first")
args = parser.parse_args()

target = os.path.join(os.environ["HF_LEROBOT_HOME"], args.repo_id)
if os.path.exists(target):
    if not args.overwrite:
        sys.exit(f"{target} already exists; pass --overwrite to replace it")
    # Never delete a dataset (CLAUDE.md rule 3): move the old one to $HF_LEROBOT_HOME/_archive/.
    import time

    old = os.path.join(os.environ["HF_LEROBOT_HOME"], "_archive", f"{args.repo_id.replace('/', '_')}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(os.path.dirname(old), exist_ok=True)
    shutil.move(target, old)
    print(f"[convert] moved the old dataset to {old}", flush=True)

folders = {os.path.dirname(f) for f in args.hdf5_file}
if len(folders) != 1:
    sys.exit("all --hdf5_file files must be in the same folder")
script = os.path.join(os.environ["XLE_THIRD_PARTY"], "leisaac/scripts/convert/isaaclab2lerobotv3.py")
sys.argv = [
    script,
    "--task_name", args.task,
    "--repo_id", args.repo_id,
    "--fps", "30",
    "--hdf5_root", folders.pop(),
    "--hdf5_files", ",".join(os.path.basename(f) for f in args.hdf5_file),
]  # fmt: skip
os.environ["XLE_VIDEO_CAMERA"] = "0"  # video-only cameras are not dataset features (put_it_back_env_cfg.py)
try:
    runpy.run_path(script, run_name="__main__")
    print(f"[convert] dataset at {target}", flush=True)
except BaseException:
    import traceback

    traceback.print_exc()
    sys.stdout.flush()
    os._exit(1)  # without this a crash leaves Isaac Sim running until the job's time limit
os._exit(0)
