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
parser.add_argument("--hdf5_file", default=os.path.join(os.environ["XLE_DATA"], "hdf5", "xle_sandbox_v0.hdf5"))
parser.add_argument("--repo_id", default="local/xle_sandbox_v0")
parser.add_argument("--overwrite", action="store_true", help="delete an existing dataset with this repo_id first")
args = parser.parse_args()

target = os.path.join(os.environ["HF_LEROBOT_HOME"], args.repo_id)
if os.path.exists(target):
    if not args.overwrite:
        sys.exit(f"{target} already exists; pass --overwrite to replace it")
    shutil.rmtree(target)

script = os.path.join(os.environ["XLE_THIRD_PARTY"], "leisaac/scripts/convert/isaaclab2lerobotv3.py")
sys.argv = [
    script,
    "--task_name", "XLeRobot-Sandbox-v0",
    "--repo_id", args.repo_id,
    "--fps", "30",
    "--hdf5_root", os.path.dirname(args.hdf5_file),
    "--hdf5_files", os.path.basename(args.hdf5_file),
]  # fmt: skip
runpy.run_path(script, run_name="__main__")
print(f"[convert] dataset at {target}")
