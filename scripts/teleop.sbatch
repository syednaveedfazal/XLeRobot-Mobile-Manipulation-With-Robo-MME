#!/usr/bin/env bash
#SBATCH --job-name=xle-teleop
#SBATCH --partition=mlgpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/teleop-%j.out
# Browser teleop of XLeRobot in ManiSkill. Usually started via scripts/start_teleop.sh.
# mlgpu = A40 (has RT cores + graphics); sgpu (A100) also works with the default shader.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source setup/env.sh
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
exec python -u -m xlerobot_sim.teleop_web "$@"
