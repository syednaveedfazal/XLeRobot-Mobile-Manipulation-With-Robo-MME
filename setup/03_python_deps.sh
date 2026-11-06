#!/bin/bash
# 03_python_deps.sh - install LeIsaac, LeRobot and our package for the container's Python.
# Runs INSIDE the container (see 03_python_deps.sbatch).
#
# --user with PYTHONUSERBASE puts packages in $XLE_PYDEPS on Lustre. pip still
# sees the image's own packages, so torch and friends are reused, not re-downloaded.
#
# --use-deprecated=legacy-resolver: Isaac Sim bundles its own imageio, and the
# strict resolver rejects lerobot's imageio[ffmpeg] requirement against it
# (ResolutionImpossible). The legacy resolver installs anyway, so `pip check`
# below prints anything that really does not match.

set -euo pipefail
PIP="/isaac-sim/python.sh -m pip"
C=setup/constraints.txt

$PIP install --user --no-warn-script-location --use-deprecated=legacy-resolver -c $C -e "$XLE_THIRD_PARTY/leisaac/source/leisaac[lerobot]"
$PIP install --user --no-warn-script-location --no-deps -e .

echo "== pip check =="; $PIP check || true

$PIP freeze --user > requirements.lock.txt
/isaac-sim/python.sh -c "import torch, numpy, lerobot, leisaac, xle_leisaac; print('torch', torch.__version__, 'numpy', numpy.__version__, 'lerobot', lerobot.__version__)"
