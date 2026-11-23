#!/usr/bin/env bash
# put_it_back_pipeline.sh - queue the whole PutItBack pipeline as dependent Slurm jobs, then
# you can log out. Each job starts only when the jobs it needs have finished.
#
#   bash scripts/put_it_back_pipeline.sh            # data, then training
#   bash scripts/put_it_back_pipeline.sh --convert  # convert the recorded chunks again, then train
#   bash scripts/put_it_back_pipeline.sh --train    # training only, on the existing dataset
#
#   check    expert check, 3 episodes with videos (stops everything unless >= 2 succeed)
#   record   CHUNKS short jobs of EPISODES demos each (mlgpu_devel, start sooner)
#   convert  all chunks -> LeRobot dataset local/xle_put_it_back_v0
#   cache    memory features for pi05_mem (tools/build_memory_cache.py)
#   train    pi0.5 without memory (baseline) and with memory (framesamp_modul),
#            slurm/train_pi05_mem.sbatch, same settings otherwise
#
# Training needs Google's gated PaliGemma files. Log in to Hugging Face once, any time before
# the training jobs start (they check for the login first and stop if it is missing):
#   1. huggingface.co -> open google/paligemma-3b-pt-224 -> accept the licence
#   2. huggingface.co -> Settings -> Access Tokens -> new token, type "Read"
#   3. on Marvin:  slurm/in_container.sh /isaac-sim/python.sh -m huggingface_hub.commands.huggingface_cli login
#      paste the token (answer n to "add as git credential"); saved in $XLE_CACHE/huggingface/token
#
# Overrides (environment): CHUNKS (default 12), EPISODES per chunk (10), STEPS (training, 30000).
# Data jobs ask for little (short, 4 CPUs, 16-20 GB) so Slurm can fit them into gaps on a full cluster.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.local.sh
mkdir -p logs
REPO=local/xle_put_it_back_v0
CHUNKS=${CHUNKS:-12}
export EPISODES=${EPISODES:-10}
SMALL="--cpus-per-task=4"
TRAIN_PARTITION=${TRAIN_PARTITION:-mlgpu_medium}  # 1-day limit; 8 h is too short for 30k pi0.5 steps
TRAIN_TIME=${TRAIN_TIME:-24:00:00}
EXCLUDE=${EXCLUDE:---exclude=mlgpu024}  # a node that hung in Prolog before

DEP=""
if [ "${1:-}" = "--convert" ]; then
    CONVERT=$(sbatch --parsable $EXCLUDE --partition=mlgpu_short --time=08:00:00 slurm/put_it_back_data.sbatch convert)
    echo "convert  job $CONVERT   log: logs/put_it_back_data_$CONVERT.out"
    DEP="--dependency=afterok:$CONVERT"
elif [ "${1:-}" != "--train" ]; then
    CHECK=$(sbatch --parsable $EXCLUDE $SMALL --mem=16G --time=00:15:00 slurm/put_it_back_data.sbatch check)
    echo "check    job $CHECK   log: logs/put_it_back_data_$CHECK.out"
    RECORDS=()
    for c in $(seq 1 "$CHUNKS"); do
        RECORDS+=("$(sbatch --parsable $EXCLUDE $SMALL --mem=20G --time=00:40:00 --dependency=afterok:$CHECK \
            slurm/put_it_back_data.sbatch record "$c")")
    done
    echo "record   jobs ${RECORDS[*]}  ($CHUNKS x $EPISODES episodes)"
    CONVERT=$(sbatch --parsable $EXCLUDE --dependency=afterany:$(IFS=:; echo "${RECORDS[*]}") \
        --partition=mlgpu_short --time=08:00:00 slurm/put_it_back_data.sbatch convert)
    echo "convert  job $CONVERT   log: logs/put_it_back_data_$CONVERT.out"
    DEP="--dependency=afterok:$CONVERT"
fi

CACHE=$(XLE_WITH_PI=1 sbatch --parsable $EXCLUDE $DEP --output=logs/put_it_back_cache_%j.out slurm/sim_job.sbatch \
    /isaac-sim/python.sh tools/build_memory_cache.py --repo_id "$REPO")
echo "cache    job $CACHE   log: logs/put_it_back_cache_$CACHE.out"
BASE=$(REPO_ID=$REPO sbatch --parsable $EXCLUDE $DEP --partition="$TRAIN_PARTITION" --time="$TRAIN_TIME" \
    --output=logs/put_it_back_train_none_%j.out slurm/train_pi05_mem.sbatch none)
echo "train    job $BASE   (no memory)    log: logs/put_it_back_train_none_$BASE.out"
MEM=$(REPO_ID=$REPO sbatch --parsable $EXCLUDE --dependency=afterok:$CACHE --partition="$TRAIN_PARTITION" \
    --time="$TRAIN_TIME" --output=logs/put_it_back_train_mem_%j.out slurm/train_pi05_mem.sbatch framesamp_modul)
echo "train    job $MEM   (with memory)  log: logs/put_it_back_train_mem_$MEM.out"

if [ ! -f "$HF_HOME/token" ]; then
    echo
    echo "NOTE: not logged in to Hugging Face yet. Do it before the training jobs start (see the top of this script):"
    echo "  slurm/in_container.sh /isaac-sim/python.sh -m huggingface_hub.commands.huggingface_cli login"
fi
echo
echo "Check progress any time with:  squeue -u \$USER   and the logs above. Checkpoints: \$XLE_WORK/train_runs/"
