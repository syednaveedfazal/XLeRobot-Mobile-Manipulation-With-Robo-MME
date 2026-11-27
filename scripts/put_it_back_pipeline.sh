#!/usr/bin/env bash
# put_it_back_pipeline.sh - queue the whole PutItBack pipeline as dependent Slurm jobs, then you
# can log out. Each job starts only when the jobs it needs have finished.
#
#   bash scripts/put_it_back_pipeline.sh            # check -> record -> convert -> merge -> cache + train both
#   bash scripts/put_it_back_pipeline.sh --convert  # convert + merge the existing recordings, then train
#   bash scripts/put_it_back_pipeline.sh --train    # training only, on the existing merged dataset
#
#   check    expert check, 6 noisy episodes with videos (stops everything unless >= 5 succeed, none late)
#   record   CHUNKS short jobs of EPISODES demos (2 of every 3 with recovery noise; stations cycled)
#   convert  one job per chunk after its recording, at most 6 at a time (the cluster disk is nearly
#            full; 15 at once ran it out of space in run 2) -> <REPO>_c<N>; the raw chunk is deleted then
#   merge    all chunks -> <REPO> (tools/merge_datasets.py), stops if the turn-back time gives the
#            station away (B12); then deletes the per-chunk datasets and the dataset in REPLACES
#   cache    memory features for pi05_mem (tools/build_memory_cache.py)
#   train    pi0.5 without memory (baseline) and with memory (framesamp_modul), same settings otherwise;
#            a resume job continues a run that stops early (time limit, node failure)
#   eval     EVAL_EPISODES closed-loop episodes per model, seed 777 (tools/eval_put_it_back.py)
#
# Training needs Google's gated PaliGemma files. Log in to Hugging Face once, any time before
# the training jobs start (they check for the login first and stop if it is missing):
#   1. huggingface.co -> open google/paligemma-3b-pt-224 -> accept the licence
#   2. huggingface.co -> Settings -> Access Tokens -> new token, type "Read"
#   3. on Marvin:  slurm/in_container.sh /isaac-sim/python.sh -m huggingface_hub.commands.huggingface_cli login
#
# Overrides (environment): CHUNKS (20), EPISODES per chunk (9: a multiple of 3 keeps stations balanced),
# REPO (local/xle_put_it_back_v2), TAG (put_it_back_v2), REPLACES (local/xle_put_it_back_v1, deleted after
# a good merge; REPLACES= keeps it), STEPS (12000), SAVE_FREQ (6000: 2 checkpoints of ~44 GB per run),
# TRAIN_PARTITION (A40/A100 long+medium), TRAIN_TIME (22:00:00), EVAL_EPISODES (50).
# Data jobs ask for little (short, 4 CPUs, 16-20 GB) so Slurm can fit them into gaps on a full cluster.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.local.sh
mkdir -p logs
export REPO=${REPO:-local/xle_put_it_back_v2} TAG=${TAG:-put_it_back_v2}
export REPLACES=${REPLACES-local/xle_put_it_back_v1}
CHUNKS=${CHUNKS:-20}
export EPISODES=${EPISODES:-9}
STEPS=${STEPS:-12000}
SAVE_FREQ=${SAVE_FREQ:-6000}
TRAIN_PARTITION=${TRAIN_PARTITION:-mlgpu_long,mlgpu_medium,sgpu_long,sgpu_medium}  # 12000 steps x ~5 s = ~17 h; A40 or A100 (no Isaac rendering), whichever starts first
TRAIN_TIME=${TRAIN_TIME:-22:00:00}
EVAL_EPISODES=${EVAL_EPISODES:-50}
SMALL="--cpus-per-task=4"
EXCLUDE=${EXCLUDE:---exclude=mlgpu024}  # a node that hung in Prolog before

DEP=""
if [ "${1:-}" != "--train" ]; then
    CONVERTS=()
    if [ "${1:-}" = "--convert" ]; then
        for f in "$XLE_DATA"/hdf5/${TAG}_c*.hdf5; do
            c=${f##*_c}; c=${c%.hdf5}
            AFTER=""; [ ${#CONVERTS[@]} -ge 6 ] && AFTER="--dependency=afterany:${CONVERTS[-6]}"
            CONVERTS+=("$(sbatch --parsable $EXCLUDE $SMALL --mem=24G --partition=mlgpu_short,mlgpu_medium,mlgpu_long --time=02:00:00 $AFTER \
                slurm/put_it_back_data.sbatch convert "$c")")
        done
    else
        CHECK=$(sbatch --parsable $EXCLUDE $SMALL --mem=16G --time=00:50:00 slurm/put_it_back_data.sbatch check)
        echo "check    job $CHECK   log: logs/put_it_back_data_$CHECK.out"
        for c in $(seq 1 "$CHUNKS"); do
            R=$(NOISE=$(( c % 3 != 0 )) sbatch --parsable $EXCLUDE $SMALL --mem=20G --time=00:55:00 \
                --dependency=afterok:$CHECK slurm/put_it_back_data.sbatch record "$c")
            DEPS="afterok:$R"; [ ${#CONVERTS[@]} -ge 6 ] && DEPS="$DEPS,afterany:${CONVERTS[-6]}"
            CONVERTS+=("$(sbatch --parsable $EXCLUDE $SMALL --mem=24G --partition=mlgpu_short,mlgpu_medium,mlgpu_long --time=02:00:00 \
                --dependency=$DEPS slurm/put_it_back_data.sbatch convert "$c")")
        done
        echo "record + convert: $CHUNKS chunks x $EPISODES episodes"
    fi
    MERGE=$(sbatch --parsable $EXCLUDE --dependency=afterany:$(IFS=:; echo "${CONVERTS[*]}") \
        --partition=mlgpu_short --time=02:00:00 slurm/put_it_back_data.sbatch merge)
    echo "merge    job $MERGE   log: logs/put_it_back_data_$MERGE.out"
    DEP="--dependency=afterok:$MERGE"
fi

TRAIN_ARGS="--num_workers=12 --log_freq=100 --save_freq=$SAVE_FREQ"
CACHE=$(XLE_WITH_PI=1 sbatch --parsable $EXCLUDE $DEP --partition=mlgpu_short,sgpu_short,mlgpu_medium,sgpu_medium --time=02:00:00 --mem=48G \
    --output=logs/put_it_back_cache_%j.out slurm/sim_job.sbatch \
    /isaac-sim/python.sh tools/build_memory_cache.py --repo_id "$REPO" --pad_black 1)
echo "cache    job $CACHE   log: logs/put_it_back_cache_$CACHE.out"
BASE=$(REPO_ID=$REPO STEPS=$STEPS BATCH=16 GPU_LOG=1 EXTRA="$TRAIN_ARGS" sbatch --parsable $EXCLUDE $DEP \
    --partition="$TRAIN_PARTITION" --time="$TRAIN_TIME" --cpus-per-task=10 --mem=80G \
    --output=logs/put_it_back_train_none_%j.out slurm/train_pi05_mem.sbatch none)
echo "train    job $BASE   (no memory)    log: logs/put_it_back_train_none_$BASE.out"
# The memory run keeps the feature cache in RAM (~64 KB per frame: ~35 GB for ~500k frames).
MEM=$(REPO_ID=$REPO STEPS=$STEPS BATCH=16 GPU_LOG=1 EXTRA="$TRAIN_ARGS" sbatch --parsable $EXCLUDE \
    --dependency=afterok:$CACHE --partition="$TRAIN_PARTITION" --time="$TRAIN_TIME" --cpus-per-task=10 --mem=128G \
    --output=logs/put_it_back_train_mem_%j.out slurm/train_pi05_mem.sbatch framesamp_modul)
echo "train    job $MEM   (with memory)  log: logs/put_it_back_train_mem_$MEM.out"

# If a training job stops early (time limit, node failure), a resume job continues it from its last
# checkpoint (it never runs when training finished). Then the closed-loop evaluation, same seed for both.
V=$(basename "$REPO"); V=${V##*_}
for pair in "none:$BASE:80G" "framesamp_modul:$MEM:128G"; do
    IFS=: read -r MEMORY JOB RAM <<< "$pair"
    RUN_DIR=$XLE_WORK/train_runs/pi05_${MEMORY}_$(basename "$REPO")_$JOB
    SHORT=$([ "$MEMORY" = none ] && echo none || echo mem)
    RESUME=$(RESUME_RUN=$RUN_DIR STEPS=$STEPS sbatch --parsable $EXCLUDE --dependency=afternotok:$JOB --kill-on-invalid-dep=yes \
        --partition="$TRAIN_PARTITION" --time="$TRAIN_TIME" --cpus-per-task=10 --mem=$RAM \
        --output=logs/put_it_back_resume_${SHORT}_%j.out slurm/train_pi05_mem.sbatch "$MEMORY")
    for after in "afterok:$JOB" "afterok:$RESUME"; do  # one of the two runs, the other never starts
        E=$(XLE_WITH_PI=1 sbatch --parsable $EXCLUDE --dependency=$after --kill-on-invalid-dep=yes \
            --partition=mlgpu_short,mlgpu_medium,mlgpu_long --time=08:00:00 --output=logs/put_it_back_eval_${SHORT}_%j.out slurm/sim_job.sbatch \
            /isaac-sim/python.sh tools/eval_put_it_back.py --checkpoint "$RUN_DIR/checkpoints/last/pretrained_model" \
            --name "${V}_${SHORT}_$((STEPS / 1000))k" --episodes "$EVAL_EPISODES" --seed 777 --videos 4)
        echo "eval     job $E   ($SHORT, $after)  log: logs/put_it_back_eval_${SHORT}_$E.out"
    done
    echo "resume   job $RESUME   ($SHORT, only if training stops early)"
done

if [ ! -f "$HF_HOME/token" ]; then
    echo
    echo "NOTE: not logged in to Hugging Face yet. Do it before the training jobs start (see the top of this script)."
fi
echo
echo "Check progress any time with:  squeue -u \$USER   and the logs above. Checkpoints: \$XLE_WORK/train_runs/"
