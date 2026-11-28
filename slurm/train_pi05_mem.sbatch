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

if [ ! -f "$HF_HOME/token" ] && [ -z "${HF_TOKEN:-}" ]; then
    echo "[train] not logged in to Hugging Face (no $HF_HOME/token): pi0.5 needs the gated"
    echo "        google/paligemma-3b-pt-224 tokenizer. Accept its licence on huggingface.co, then run"
    echo "        slurm/in_container.sh /isaac-sim/python.sh -m huggingface_hub.commands.huggingface_cli login"
    exit 1
fi

MEMORY=${1:?usage: sbatch slurm/train_pi05_mem.sbatch none|framesamp_modul}
REPO_ID=${REPO_ID:-local/xle_sandbox_v0}
CAMERA=${CAMERA:-observation.images.head}
PRETRAINED=${PRETRAINED:-lerobot/pi05_base}
# lerobot/pi05_base revision before 2026-06-03, when its processor files started naming a
# "relative_actions_processor" step that our LeRobot (0.4.2) does not have. The weights and
# config.json are byte-identical to later revisions. Set PRETRAINED_REVISION= to use the latest.
PRETRAINED_REVISION=${PRETRAINED_REVISION-a538eb273274eb30f126a118f39dbc0ee212c883}
if [ -n "$PRETRAINED_REVISION" ] && [ ! -d "$PRETRAINED" ]; then
    PRETRAINED=$(XLE_WITH_PI=1 slurm/in_container.sh /isaac-sim/python.sh -c \
        "from huggingface_hub import snapshot_download; print(snapshot_download('$PRETRAINED', revision='$PRETRAINED_REVISION'))" \
        2>/dev/null | tail -1)
    [ -d "$PRETRAINED" ] || { echo "[train] could not download the pinned pi05_base revision"; exit 1; }
fi
STEPS=${STEPS:-30000}
BATCH=${BATCH:-16}
RUN=pi05_${MEMORY}_$(basename "$REPO_ID")_${SLURM_JOB_ID}

ARGS=(
    --dataset.repo_id="$REPO_ID"
    --dataset.video_backend=pyav  # torchcodec needs system FFmpeg libraries the container lacks
    --policy.pad_black=${PAD_BLACK:-true}  # letterbox bars black, not -3 (configuration.py); PAD_BLACK=false = runs before 2026-11-24
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
if [ "$MEMORY" != none ] && [ -z "${RESUME_RUN:-}" ]; then  # a resume reads its cache path from train_config.json
    FEATURES="$HF_LEROBOT_HOME/$REPO_ID/memory_features/${CAMERA##*.}"
    [ -f "$FEATURES/features.npy" ] || { echo "no memory cache in $FEATURES (tools/build_memory_cache.py)"; exit 1; }
    ARGS+=(--policy.memory_camera="$CAMERA" --policy.memory_feature_dir="$FEATURES")
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "[train] $RUN"
if [ "${GPU_LOG:-0}" = "1" ]; then  # GPU use, memory, power, clock, temperature every 15 s
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,temperature.gpu,clocks_throttle_reasons.active \
        --format=csv -l 15 > "logs/gpu_${SLURM_JOB_ID}.csv" &
fi
# RESUME_RUN=<run dir>: continue that run from its last checkpoint (e.g. after an 8 h time limit);
# exits at once if it already reached STEPS.
if [ -n "${RESUME_RUN:-}" ]; then
    LAST=$(readlink "$RESUME_RUN/checkpoints/last" 2>/dev/null || true)
    if [ "$LAST" = "$(printf %06d "$STEPS")" ]; then echo "[train] $RESUME_RUN already finished (step $LAST)"; exit 0; fi
    [ -n "$LAST" ] || { echo "[train] no checkpoint to resume in $RESUME_RUN"; exit 1; }
    # Only the last checkpoint is needed to go on (~44 GB each); delete the older ones.
    for c in "$RESUME_RUN"/checkpoints/[0-9]*; do
        [ "$(basename "$c")" = "$LAST" ] || { rm -rf "$c"; echo "[train] deleted the older checkpoint $(basename "$c")"; }
    done
    echo "[train] resuming $RESUME_RUN from step $LAST"
    exec slurm/in_container.sh /isaac-sim/python.sh tools/train_policy.py \
        --config_path="$RESUME_RUN/checkpoints/last/pretrained_model/train_config.json" --resume=true
fi
slurm/in_container.sh /isaac-sim/python.sh tools/train_policy.py "${ARGS[@]}" ${EXTRA:-}
