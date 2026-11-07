#!/bin/bash
#SBATCH --job-name=xle-teleop-isaac
#SBATCH --partition=mlgpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/teleop-%j.out
# Browser teleop of XLeRobot in Isaac Lab (the twin of scripts/teleop.sbatch for ManiSkill).
# Usually started via scripts/start_teleop_isaac.sh, which reads this job's log for the URL.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
exec slurm/in_container.sh /isaac-sim/python.sh -m xle_leisaac.teleop.isaac_teleop_web "$@"
