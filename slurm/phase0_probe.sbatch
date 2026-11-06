#!/bin/bash
# phase0_probe.sbatch - read-only look at an A40 node. Installs nothing.
#
# Why: the login node has no GPU, so the facts that decide the install route
# (driver version, glibc, Vulkan, internet on compute nodes) can only be read
# from inside a job. Isaac Sim 5.1 needs driver >= 580.65.06 and glibc >= 2.35
# for a native install.
#
# Run it with:
#   sbatch --output=$HOME/xle_probe_%j.out slurm/phase0_probe.sbatch
#
#SBATCH --job-name=xle_probe
#SBATCH --partition=mlgpu_devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00

echo "== host ==";    hostname; grep PRETTY_NAME /etc/os-release; uname -r
echo "== glibc ==";   ldd --version | head -1
echo "== gpu ==";     nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
nvidia-smi | head -4
echo "== encoder =="; nvidia-smi -q | grep -i -A2 encoder | head -12

# Isaac Sim renders through Vulkan, so the NVIDIA graphics libraries and the
# Vulkan ICD file must exist on the node, not only the CUDA driver.
echo "== graphics libs =="
ldconfig -p | grep -E 'libGLX_nvidia|libEGL_nvidia|libnvidia-glcore|libvulkan|libnvidia-encode'
ls -l /etc/vulkan/icd.d /usr/share/vulkan/icd.d 2>&1
which vulkaninfo || echo "no vulkaninfo on node"

echo "== internet =="
for url in https://pypi.nvidia.com https://nvcr.io/v2/ https://github.com; do
    printf '%s -> ' "$url"; curl -sI --max-time 15 "$url" | head -1 || true; echo
done

echo "== tools ==";   apptainer --version
echo "== local disk =="; echo "TMPDIR=$TMPDIR"; df -h /tmp
