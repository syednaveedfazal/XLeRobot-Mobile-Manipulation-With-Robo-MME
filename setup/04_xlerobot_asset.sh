#!/bin/bash
# 04_xlerobot_asset.sh - build the XLeRobot robot asset used in the sim.
#
# 1. Unpack XLeRobot's own model (body, cart, neck, head, virtual base joints).
# 2. Download the official SO-101 arm model (URDF + meshes) at a pinned commit.
# 3. Merge them into one URDF (plain Python, login node is fine).
# 4. Submit a GPU job that converts it to USD and builds the sandbox props.
#
# Run from the repo root:  bash setup/04_xlerobot_asset.sh

set -euo pipefail
source env.local.sh

mkdir -p "$XLE_ASSETS/robots/xlerobot"
unzip -q -o "$XLE_THIRD_PARTY/XLeRobot/simulation/Isaac_sim/xlerobot.zip" -d "$XLE_ASSETS/robots/xlerobot"

SO101_COMMIT=a758567c3978dfeefe282ede0500085a48fe8f78
SO101_DIR="$XLE_THIRD_PARTY/so_arm100_urdf"
RAW="https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/$SO101_COMMIT/Simulation/SO101"
mkdir -p "$SO101_DIR/assets"
for f in so101_new_calib.urdf so101_new_calib_camera.urdf; do
    curl -fsSL -o "$SO101_DIR/$f" "$RAW/$f"
done
for f in $(grep -o 'assets/[^"]*\.stl' "$SO101_DIR/so101_new_calib_camera.urdf" | sort -u); do
    curl -fsSL -o "$SO101_DIR/$f" "$RAW/$f"
done
echo "$SO101_COMMIT" > "$SO101_DIR/COMMIT"

python3 tools/build_xlerobot_urdf.py

# Converting to USD and building the sandbox props start Isaac Sim, so they run as a GPU job.
sbatch --output="$HOME/xle_asset_%j.out" slurm/sim_job.sbatch bash -c \
    "/isaac-sim/python.sh tools/build_xlerobot_usd.py && /isaac-sim/python.sh tools/build_props_usd.py"
