#!/bin/bash
# in_container.sh - run one command inside the Isaac Lab container.
#
# Every other script goes through this file, so it is the place to read if you
# want to know how the container is set up.
#
#   slurm/in_container.sh /isaac-sim/python.sh --version
#   slurm/in_container.sh bash          # a shell inside the container
#
# The .sif image is read-only, so anything Isaac Sim writes must be pointed at
# the workspace with a bind (-B host_path:container_path).

set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$REPO/env.local.sh"

# --nv            bring in the host's NVIDIA driver libraries
# --containall    hide the host's home, /tmp and environment for a clean run
# --writable-tmpfs small in-memory layer for stray writes
# PYTHONUNBUFFERED: Kit exits abruptly, which loses buffered print() output
# --home          container $HOME on Lustre, so ~/.cache/ov, ~/.nv etc. stay
#                 out of the real $HOME
exec apptainer exec --nv --containall --writable-tmpfs \
    --home "$XLE_CONTAINER_HOME:/home/$USER" \
    -B "${TMPDIR:-/tmp}:/tmp" \
    -B "$XLE_WORK" \
    -B "$REPO" \
    -B "$XLE_CACHE/kit/cache:/isaac-sim/kit/cache" \
    -B "$XLE_CACHE/kit/data:/isaac-sim/kit/data" \
    -B "$XLE_CACHE/kit/logs:/isaac-sim/kit/logs" \
    --env OMNI_KIT_ACCEPT_EULA=YES,ACCEPT_EULA=Y,PRIVACY_CONSENT=Y \
    --env PYTHONUSERBASE="$XLE_PYDEPS",PYTHONUNBUFFERED=1 \
    --env PATH="$XLE_PYDEPS/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    --env PIP_CACHE_DIR="$PIP_CACHE_DIR",HF_HOME="$HF_HOME",TORCH_HOME="$TORCH_HOME" \
    --env LEISAAC_ASSETS_ROOT="$LEISAAC_ASSETS_ROOT",XLE_WORK="$XLE_WORK",XLE_LOGS="$XLE_LOGS" \
    --env XLE_THIRD_PARTY="$XLE_THIRD_PARTY",XLE_DATA="$XLE_DATA",XLE_ASSETS="$XLE_ASSETS" \
    --pwd "$REPO" \
    "$XLE_SIF" "$@"
