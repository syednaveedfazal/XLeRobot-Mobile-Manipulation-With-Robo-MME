#!/bin/bash
# 01_pull_image.sbatch - download the Isaac Lab container and convert it to a .sif.
#
# Why this image: LeIsaac pins Isaac Lab v2.3.0, and NVIDIA's isaac-lab:2.3.0
# image ships Isaac Lab on top of Isaac Sim, so one file gives us both.
# Why a job: the conversion (mksquashfs) is CPU-heavy, which is not allowed on
# login nodes. It needs no GPU, so it runs on a CPU partition.
#
# Run from the repo root, after creating env.local.sh:
#   sbatch --output=$HOME/xle_pull_%j.out setup/01_pull_image.sbatch
#
#SBATCH --job-name=xle_pull
#SBATCH --partition=intelsr_short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
source "$SLURM_SUBMIT_DIR/env.local.sh"

# Unpack on the node's local disk: tens of thousands of small files are slow on Lustre.
export APPTAINER_TMPDIR=${TMPDIR:-/tmp}/xle_apptainer_$SLURM_JOB_ID
mkdir -p "$APPTAINER_TMPDIR"
trap 'rm -rf "$APPTAINER_TMPDIR"' EXIT
df -h "$APPTAINER_TMPDIR"

apptainer pull "$XLE_IMAGES/isaac-lab_2.3.0.sif" docker://nvcr.io/nvidia/isaac-lab:2.3.0
ls -lh "$XLE_IMAGES"
