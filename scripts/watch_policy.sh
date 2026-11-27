#!/usr/bin/env bash
# watch_policy.sh - watch a trained PutItBack policy drive the robot live in your browser.
#
#   bash scripts/watch_policy.sh none            # pi0.5 without memory (last checkpoint)
#   bash scripts/watch_policy.sh memory          # pi0.5 with memory
#   bash scripts/watch_policy.sh memory 006000   # an earlier checkpoint
#   bash scripts/watch_policy.sh none last --seed 777   # extra args go to isaac_teleop_web.py
#
# Same page as the browser teleop (scripts/start_teleop_isaac.sh): a chase camera, the head and
# both wrist cameras, live. The status line shows the episode, which station the block started
# on, and how the last episodes ended. Episodes end as in training (success, wrong station,
# dropped, or 90 s) and the next one starts by itself; "Reset" in the page starts a new one.
# Runs 1 h on mlgpu_devel by default: PARTITION=mlgpu_short TIME=03:00:00 for longer.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.local.sh
KIND=${1:?usage: bash scripts/watch_policy.sh none|memory [checkpoint step, default last] [extra args]}
STEP=${2:-last}
shift $(( $# >= 2 ? 2 : 1 ))
case "$KIND" in
    none) PATTERN=pi05_none_xle_put_it_back_v0_* ;;
    memory) PATTERN=pi05_framesamp_modul_xle_put_it_back_v0_* ;;
    *) echo "first argument must be none or memory"; exit 2 ;;
esac
RUN=$(ls -d "$XLE_WORK"/train_runs/$PATTERN 2>/dev/null | sort | tail -1)
CKPT=$RUN/checkpoints/$STEP/pretrained_model
[ -f "$CKPT/model.safetensors" ] || { echo "no checkpoint at $CKPT"; exit 1; }
echo "policy: $KIND, checkpoint $CKPT"
export XLE_WITH_PI=1 SBATCH_SCRIPT=slurm/teleop_isaac.sbatch SBATCH_EXTRA="--mem=48G"
exec bash scripts/start_teleop.sh --task XLeRobot-PutItBack-v0 --policy "$CKPT" "$@"
