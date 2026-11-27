#!/bin/bash
# put_it_back_data.sbatch - PutItBack demonstrations in short jobs (they start sooner on a
# full cluster). Usually submitted by scripts/put_it_back_pipeline.sh, one job per stage:
#
#   sbatch slurm/put_it_back_data.sbatch check       # 6 episodes, 2 per station, with recovery noise and videos
#                                                    # ($XLE_LOGS/put_it_back_expert/); fails (so nothing after it
#                                                    # runs) unless >= 5 succeed and none reaches the table late (B12)
#   sbatch slurm/put_it_back_data.sbatch record N    # chunk N: EPISODES episodes, seed N*1000,
#                                                    # to $XLE_DATA/hdf5/<TAG>_cN.hdf5 (+ _episodes.json)
#   sbatch slurm/put_it_back_data.sbatch convert N   # chunk N -> LeRobot dataset <REPO>_cN (successful episodes);
#                                                    # then deletes the chunk's raw HDF5 (KEEP_HDF5=1 keeps it)
#   sbatch slurm/put_it_back_data.sbatch merge       # all <REPO>_c* -> <REPO> (tools/merge_datasets.py);
#                                                    # fails below MIN_DEMOS or if the turn-back time predicts the
#                                                    # station (B12); then deletes the raw chunks and parts, and the
#                                                    # dataset it replaces (REPLACES=local/..., if set)
#
# Converting one big recording is slow (one process, ~17 frames/s), so chunks are converted in
# parallel and merged. Recording cycles the cube's station 0, 1, 2, ... so demos are balanced.
# Overrides (environment): EPISODES per chunk (default 9), MIN_DEMOS (default 60), NOISE=1 for
# recovery demos (action_noise.py), REPO (default local/xle_put_it_back_v2), TAG (default put_it_back_v2).
#
#SBATCH --job-name=xle_pib_data
#SBATCH --partition=mlgpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:55:00
#SBATCH --output=logs/put_it_back_data_%j.out

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source env.local.sh
STAGE=${1:?usage: sbatch slurm/put_it_back_data.sbatch check | record N | convert N | merge}
EPISODES=${EPISODES:-9}
MIN_DEMOS=${MIN_DEMOS:-60}
REPO=${REPO:-local/xle_put_it_back_v2}
TAG=${TAG:-put_it_back_v2}
HDF5_DIR=$XLE_DATA/hdf5
KEEP='\[expert\]|\[convert\]|\[merge\]|Saving episode|Traceback|^  File|^    |^[A-Za-z]*Error'
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$HDF5_DIR"
archive() {  # keep things that may matter: move them to an _archive folder next to them
    for f in "$@"; do
        [ -e "$f" ] || continue
        mkdir -p "$(dirname "$f")/_archive/$STAMP"
        mv "$f" "$(dirname "$f")/_archive/$STAMP/"
    done
}
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

case "$STAGE" in
check)
    slurm/in_container.sh /isaac-sim/python.sh tools/build_props_usd.py 2>&1 | grep -E '\[props\]|Error'  # props in sync with the code
    archive "$XLE_LOGS/put_it_back_expert"
    slurm/in_container.sh /isaac-sim/python.sh tools/run_put_it_back_expert.py --episodes 6 --seed 1000 --videos \
        --noise --balanced 2>&1 | grep --line-buffered -E "$KEEP" | tee "$XLE_LOGS/put_it_back_check.txt"
    OK=$(grep -oE 'success ([0-9]+)/6' "$XLE_LOGS/put_it_back_check.txt" | grep -oE '[0-9]+' | head -1)
    LATE=$(grep -oE 'late episodes: [0-9]+' "$XLE_LOGS/put_it_back_check.txt" | grep -oE '[0-9]+$')
    if [ "${OK:-0}" -lt 5 ] || [ "${LATE:-1}" != 0 ]; then
        echo "##### STOP: the expert succeeded ${OK:-0}/6 times, ${LATE:-?} late at the table. Not recording. See the videos."
        exit 1
    fi
    echo "##### expert check passed (${OK}/6, none late)"
    ;;
record)
    CHUNK=${2:?record needs a chunk number}
    OUT=$HDF5_DIR/${TAG}_c$CHUNK.hdf5
    archive "$OUT" "${OUT%.hdf5}_episodes.json"
    NOISE_ARG=$([ "${NOISE:-0}" = 1 ] && echo --noise || true)
    slurm/in_container.sh /isaac-sim/python.sh tools/run_put_it_back_expert.py --episodes "$EPISODES" \
        --seed $((CHUNK * 1000)) --record "$OUT" $NOISE_ARG 2>&1 | grep --line-buffered -E "$KEEP"
    [ -s "$OUT" ] || { echo "##### no HDF5 written"; exit 1; }
    echo "##### chunk $CHUNK done: $OUT"
    ;;
convert)
    CHUNK=${2:?convert needs a chunk number}
    IN=$HDF5_DIR/${TAG}_c$CHUNK.hdf5
    [ -s "$IN" ] || { echo "##### no recording $IN"; exit 1; }
    slurm/in_container.sh /isaac-sim/python.sh tools/convert_to_lerobot.py --task XLeRobot-PutItBack-v0 \
        --hdf5_file "$IN" --repo_id "${REPO}_c$CHUNK" --overwrite 2>&1 | grep --line-buffered -E "$KEEP" \
        | tee "$XLE_LOGS/put_it_back_convert_c$CHUNK.txt"
    DEMOS=$(grep -c 'Saving episode' "$XLE_LOGS/put_it_back_convert_c$CHUNK.txt")
    [ -f "$HF_LEROBOT_HOME/${REPO}_c$CHUNK/meta/info.json" ] || { echo "##### conversion failed"; exit 1; }
    echo "##### chunk $CHUNK: $DEMOS successful demonstrations in ${REPO}_c$CHUNK"
    # The raw recording (~0.5 MB per frame) is not needed once its conversion is complete; delete it
    # now, not after the merge, so 20 chunks never sit on the (nearly full) cluster disk at once.
    WANT=$(python3 -c "import json,sys; print(sum(e['success'] for e in json.load(open(sys.argv[1]))['episodes']))" "${IN%.hdf5}_episodes.json")
    GOT=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['total_episodes'])" "$HF_LEROBOT_HOME/${REPO}_c$CHUNK/meta/info.json")
    if [ "$GOT" != "$WANT" ]; then
        echo "##### conversion incomplete: $GOT of $WANT successful episodes; keeping $IN"; exit 1
    fi
    if [ "${KEEP_HDF5:-0}" != 1 ]; then rm -f "$IN"; echo "##### deleted the raw recording $IN (converted)"; fi
    ;;
merge)
    slurm/in_container.sh /isaac-sim/python.sh tools/merge_datasets.py --repo_id "$REPO" --hdf5_dir "$HDF5_DIR" \
        --tag "$TAG" 2>&1 | grep --line-buffered -E "$KEEP" | tee "$XLE_LOGS/put_it_back_merge.txt"
    DEMOS=$(grep -oE "$REPO: [0-9]+ episodes" "$XLE_LOGS/put_it_back_merge.txt" | grep -oE '[0-9]+ episodes' | grep -oE '[0-9]+')
    echo "##### ${DEMOS:-0} demonstrations in $REPO"
    [ "${DEMOS:-0}" -ge "$MIN_DEMOS" ] || { echo "##### STOP: fewer than $MIN_DEMOS demos"; exit 1; }
    if grep -q "STOP" "$XLE_LOGS/put_it_back_merge.txt"; then
        echo "##### STOP: timing check failed (see above); raw chunks kept, training not started"; exit 1
    fi
    # The raw chunks (~0.5 MB per frame, incl. failed episodes) and the per-chunk datasets are not
    # needed once the merged dataset exists; delete them (CLAUDE.md rule 3). The small per-episode
    # side files are kept (archived). KEEP_HDF5=1 keeps everything.
    if [ "${KEEP_HDF5:-0}" != 1 ]; then
        archive "$HDF5_DIR"/${TAG}_c*_episodes.json
        rm -f "$HDF5_DIR"/${TAG}_c*.hdf5
        rm -rf "$HF_LEROBOT_HOME/${REPO}"_c*
        echo "##### deleted the raw HDF5 chunks and the per-chunk datasets (merged dataset kept)"
    fi
    if [ -n "${REPLACES:-}" ] && [ "$REPLACES" != "$REPO" ] && [ -d "$HF_LEROBOT_HOME/$REPLACES" ]; then
        rm -rf "$HF_LEROBOT_HOME/$REPLACES"
        echo "##### deleted the dataset this one replaces: $REPLACES"
    fi
    ;;
*)
    echo "unknown stage $STAGE"; exit 2 ;;
esac
