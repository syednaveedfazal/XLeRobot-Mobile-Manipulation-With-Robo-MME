#!/usr/bin/env bash
# Browser teleop of XLeRobot in Isaac Lab. Same launcher, relay and page as the ManiSkill
# teleop (scripts/start_teleop.sh); only the simulator job is different.
#   bash scripts/start_teleop_isaac.sh                                    # 1 h on mlgpu_devel
#   PARTITION=mlgpu_short TIME=04:00:00 bash scripts/start_teleop_isaac.sh
# Isaac Sim takes a few minutes to start; the script waits until the scene is ready.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SBATCH_SCRIPT=slurm/teleop_isaac.sbatch exec bash scripts/start_teleop.sh "$@"
