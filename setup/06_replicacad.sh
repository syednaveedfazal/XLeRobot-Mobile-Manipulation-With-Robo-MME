#!/bin/bash
# 06_replicacad.sh - convert a ReplicaCAD apartment to USD for the Isaac Lab env XLeRobot-ReplicaCAD-v0.
#
# Needs Part B's setup (setup/install.sh): its conda env has trimesh, and ReplicaCAD is already
# downloaded into the ManiSkill workspace. The conversion is plain Python on the login node
# (~30 s); at the end a short GPU job lets the objects settle once (tools/settle_scene.py).
#
#   bash setup/06_replicacad.sh            # apt_0 (the layout ManiSkill uses by default)
#   bash setup/06_replicacad.sh apt_3      # another furniture layout
#
# Output: $XLE_ASSETS/replicacad/ (~60 MB per apartment). See tools/build_replicacad_usd.py for the layout.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SCENE=${1:-apt_0}

source setup/env.sh                     # conda env with trimesh; sets MS_ASSET_DIR
python -c "import pxr" 2>/dev/null || pip install --quiet "usd-core==24.5"   # same USD as Isaac Sim 5.1
XLE_ASSETS=$(bash -c 'source env.local.sh >/dev/null && echo "$XLE_ASSETS"')

# Lustre: new files keep their first 256 KB on a nearly full metadata server; store them only
# on the object servers instead (see README, "Lustre").
mkdir -p "$XLE_ASSETS/replicacad"
lfs setstripe -c 1 "$XLE_ASSETS/replicacad" 2>/dev/null || true

XLE_ASSETS="$XLE_ASSETS" python tools/build_replicacad_usd.py --scene "$SCENE"

# Let the objects come to rest once in Isaac Sim and store those poses as their start poses.
if [ "$SCENE" = "apt_0" ]; then
    sbatch --output="$HOME/xle_settle_%j.out" slurm/sim_job.sbatch \
        /isaac-sim/python.sh tools/settle_scene.py --enable_cameras --device cpu
else
    echo "Only apt_0 is registered as an env; register $SCENE in xle_leisaac/__init__.py, then run"
    echo "tools/settle_scene.py --task <its id> --enable_cameras --device cpu in a GPU job."
fi
