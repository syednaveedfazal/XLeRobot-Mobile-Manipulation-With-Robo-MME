#!/bin/bash
# train_pi05_mem.sbatch - fine-tune pi0.5 with the memory flag; all other settings are
# identical between runs, so the no-memory baseline and memory variants compare cleanly.
#
#   sbatch slurm/train_pi05_mem.sbatch none              # baseline: current frame only
#   sbatch slurm/train_pi05_mem.sbatch framesamp_modul   # RoboMME FrameSamp + Modul
#
# Before a memory run, build the feature cache once per dataset (same weights as below):
#   XLE_WITH_PI=1 sbatch --output=$HOME/xle_cache_%j.out slurm/sim_job.sbatch \
#       /isaac-sim/python.sh tools/build_memory_cache.py --repo_id local/xle_sandbox_v0
#
# Overrides (environment): REPO_ID, CAMERA, PRETRAINED, STEPS, BATCH, EXTRA (more
# lerobot-train flags), e.g.  STEPS=2000 BATCH=8 sbatch slurm/train_pi05_mem.sbatch none
# Needs the gated google/paligemma-3b-pt-224 tokenizer: accept its licence on Hugging Face
# and `huggingface-cli login` with HF_HOME=$XLE_CACHE/huggingface.
#
#SBATCH --job-name=xle_pi05_mem
#SBATCH --partition=mlgpu_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source env.local.sh
export XLE_WITH_PI=1

MEMORY=${1:?usage: sbatch slurm/train_pi05_mem.sbatch none|framesamp_modul}
REPO_ID=${REPO_ID:-local/xle_sandbox_v0}
CAMERA=${CAMERA:-observation.images.head}
PRETRAINED=${PRETRAINED:-lerobot/pi05_base}
STEPS=${STEPS:-30000}
BATCH=${BATCH:-16}
RUN=pi05_${MEMORY}_$(basename "$REPO_ID")_${SLURM_JOB_ID}

ARGS=(
    --dataset.repo_id="$REPO_ID"
    --policy.type=pi05_mem
    --policy.memory="$MEMORY"
    --policy.pretrained_path="$PRETRAINED"
    --policy.dtype=bfloat16
    --policy.gradient_checkpointing=true
    --policy.push_to_hub=false
    --batch_size="$BATCH"
    --steps="$STEPS"
    --output_dir="$XLE_WORK/train_runs/$RUN"
    --job_name="$RUN"
    --wandb.enable=false
)
if [ "$MEMORY" != none ]; then
    FEATURES="$HF_LEROBOT_HOME/$REPO_ID/memory_features/${CAMERA##*.}"
    [ -f "$FEATURES/features.npy" ] || { echo "no memory cache in $FEATURES (tools/build_memory_cache.py)"; exit 1; }
    ARGS+=(--policy.memory_camera="$CAMERA" --policy.memory_feature_dir="$FEATURES")
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "[train] $RUN"
slurm/in_container.sh /isaac-sim/python.sh tools/train_policy.py "${ARGS[@]}" ${EXTRA:-}
