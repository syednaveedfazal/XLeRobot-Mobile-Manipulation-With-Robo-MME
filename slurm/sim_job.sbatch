#!/bin/bash
# sim_job.sbatch - run any command in the container on one A40 GPU.
#
#   sbatch --output=$HOME/xle_sim_%j.out slurm/sim_job.sbatch <command...>
#
# The partition here has a 1 h limit, which suits tests. For longer runs add
# --partition=mlgpu_short (8 h) and --time on the sbatch line.
#
#SBATCH --job-name=xle_sim
#SBATCH --partition=mlgpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:50:00

cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
slurm/in_container.sh "$@"
