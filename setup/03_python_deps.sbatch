#!/bin/bash
# 03_python_deps.sbatch - run the pip install as a CPU job. No GPU needed.
#
#   sbatch --output=$HOME/xle_deps_%j.out setup/03_python_deps.sbatch
#
#SBATCH --job-name=xle_deps
#SBATCH --partition=intelsr_short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

cd "$SLURM_SUBMIT_DIR"
slurm/in_container.sh bash setup/03_python_deps.sh
