#!/usr/bin/env bash
# One-time setup on Marvin (run on the login node; needs internet):
#   bash setup/install.sh
# Creates the workspace (if needed), a conda env in it, and downloads the ReplicaCAD scene.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! ws_find xlerobot >/dev/null 2>&1; then
    echo "[install] allocating workspace 'xlerobot' on scratch (90 days, extend with: ws_extend xlerobot 90)"
    ws_allocate -F scratch -c "XLeRobot mobile manipulation project" xlerobot 90
fi
WS="$(ws_find xlerobot)"
ENV_PREFIX="$WS/envs/xlerobot"

module load Miniforge3/24.1.2-0
eval "$(conda shell.bash hook)"
export CONDA_PKGS_DIRS="$WS/conda_pkgs" PIP_CACHE_DIR="$WS/pip_cache"

if [ ! -x "$ENV_PREFIX/bin/python" ]; then
    conda create -y -p "$ENV_PREFIX" python=3.11
fi
"$ENV_PREFIX/bin/pip" install -r "$REPO/setup/requirements.txt"

source "$REPO/setup/env.sh"
if [ ! -d "$MS_ASSET_DIR/data/scene_datasets/replica_cad_dataset" ]; then
    python -m mani_skill.utils.download_asset ReplicaCAD -y
fi
python -c "import xlerobot_sim; print('[install] ok: registered', xlerobot_sim.ENV_ID)"
echo "[install] done. Next: bash scripts/start_teleop.sh"
