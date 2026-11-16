#!/bin/bash
# 05_pi_deps.sh - extra packages for training pi0.5 (with or without memory).
#
# LeRobot's pi0.5 only runs with transformers from the `fix/lerobot_openpi` branch
# (LeRobot's `pi` extra), which needs tokenizers<0.22. The image ships transformers
# 4.57 and tokenizers 0.22, and Isaac Sim should keep those. So the two packages go
# into their own folder, $XLE_PYDEPS_PI, which is put on PYTHONPATH only when a job
# runs with XLE_WITH_PI=1 (see slurm/in_container.sh).
#
# Run from the repo root on the login node (needs internet):  bash setup/05_pi_deps.sh

set -euo pipefail
source env.local.sh

TRANSFORMERS_COMMIT=dcddb970176382c0fcf4521b0c0e6fc15894dfe0  # fix/lerobot_openpi, 2025-10-01

mkdir -p "$XLE_PYDEPS_PI"
# Lustre: new files normally keep their first 256 KB on the metadata server (Data-on-MDT).
# MDT1, where this workspace lives, refused those writes (ENOSPC, Nov 2026), so store
# files only on the object servers.
lfs setstripe -c 1 "$XLE_PYDEPS_PI" 2>/dev/null || true

slurm/in_container.sh /isaac-sim/python.sh -m pip install --no-warn-script-location --no-deps --no-compile \
    --target "$XLE_PYDEPS_PI" \
    "transformers @ git+https://github.com/huggingface/transformers.git@$TRANSFORMERS_COMMIT" \
    "tokenizers==0.21.4"

XLE_WITH_PI=1 slurm/in_container.sh /isaac-sim/python.sh -c "
from transformers.models.siglip import check
assert check.check_whether_transformers_replace_is_installed_correctly()
import transformers, tokenizers; print('[pi deps] ok: transformers', transformers.__version__, 'tokenizers', tokenizers.__version__)"
