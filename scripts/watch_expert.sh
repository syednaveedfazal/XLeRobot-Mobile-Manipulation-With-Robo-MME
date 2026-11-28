#!/usr/bin/env bash
# watch_expert.sh - watch the PutItBack scripted expert (the demo recorder) live in your browser.
#
#   bash scripts/watch_expert.sh            # exactly as it records demos
#   bash scripts/watch_expert.sh --noise    # with the recovery-demo action noise
#   bash scripts/watch_expert.sh --seed 5   # other random placements
#
# Same page as the browser teleop: chase camera, head and both wrist cameras, live. The status
# line shows the expert's current step, which station holds the cube, how big the cube is in
# pi0.5's 224 px head image right now and whether it has been clearly seen, and the results of
# the last episodes. Episodes end as in recording and the next one starts by itself.
# 1 h on mlgpu_devel by default; PARTITION=mlgpu_short TIME=03:00:00 for longer.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export SBATCH_SCRIPT=slurm/teleop_isaac.sbatch
exec bash scripts/start_teleop.sh --task XLeRobot-PutItBack-v0 --expert "$@"
