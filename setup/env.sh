# Source this file (do not execute it):   source setup/env.sh
# Activates the XLeRobot conda env on Marvin and points all heavy data at the workspace.

# Workspace (created with: ws_allocate -F scratch xlerobot 90). Override by exporting XLE_WS first.
if [ -z "${XLE_WS:-}" ]; then
    XLE_WS="$(ws_find xlerobot 2>/dev/null)"
fi
if [ -z "$XLE_WS" ] || [ ! -d "$XLE_WS" ]; then
    echo "[env.sh] workspace 'xlerobot' not found. Create it: ws_allocate -F scratch xlerobot 90" >&2
    return 1
fi
export XLE_WS

XLE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export XLE_REPO

module load Miniforge3/24.1.2-0 >/dev/null 2>&1
eval "$(conda shell.bash hook)"
export CONDA_PKGS_DIRS="$XLE_WS/conda_pkgs"
export PIP_CACHE_DIR="$XLE_WS/pip_cache"
conda activate "$XLE_WS/envs/xlerobot"

# ManiSkill assets (ReplicaCAD etc.) go to the workspace instead of ~/.maniskill
export MS_ASSET_DIR="$XLE_WS/maniskill_data"
# make `import xlerobot_sim` work from anywhere
export PYTHONPATH="$XLE_REPO${PYTHONPATH:+:$PYTHONPATH}"
