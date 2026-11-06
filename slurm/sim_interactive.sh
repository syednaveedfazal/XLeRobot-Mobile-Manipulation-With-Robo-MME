#!/bin/bash
# sim_interactive.sh - open a shell inside the container on one A40 GPU.
#
# Run from the repo root. It waits in the queue, then gives you a prompt where
# you can try commands by hand, for example:
#   /isaac-sim/python.sh tools/smoke_leisaac.py --enable_cameras

exec srun --partition=mlgpu_devel --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=00:50:00 --pty \
    slurm/in_container.sh bash
