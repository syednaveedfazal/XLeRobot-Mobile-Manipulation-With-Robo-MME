# env.example.sh - every path and setting for this project, in one place.
#
# Copy to env.local.sh (gitignored), set XLE_WORK, then load it with:
#   source env.local.sh
#
# Why: big files must stay out of $HOME (100 GB quota) and out of git, so all
# caches are pointed at a Lustre workspace. Create that with:
#   ws_allocate xle_sim 90

export XLE_WORK=/lustre/scratch/data/${USER}-xle_sim

# Layout of the work directory
export XLE_IMAGES=$XLE_WORK/images          # Apptainer .sif files
export XLE_THIRD_PARTY=$XLE_WORK/third_party # LeIsaac, XLeRobot clones
export XLE_ASSETS=$XLE_WORK/assets
export LEISAAC_ASSETS_ROOT=$XLE_ASSETS     # LeIsaac reads this; no patch needed
export XLE_DATA=$XLE_WORK/data
export XLE_LOGS=$XLE_WORK/logs
export XLE_CACHE=$XLE_WORK/cache
export XLE_SIF=$XLE_IMAGES/isaac-lab_2.3.0.sif
export XLE_PYDEPS=$XLE_WORK/pydeps           # our pip packages (the image is read-only)
export XLE_CONTAINER_HOME=$XLE_WORK/container_home # $HOME inside the container

# Tool caches
export PIP_CACHE_DIR=$XLE_CACHE/pip
export HF_HOME=$XLE_CACHE/huggingface
export HF_LEROBOT_HOME=$XLE_DATA/lerobot      # LeRobot datasets: $HF_LEROBOT_HOME/<repo_id>
export TORCH_HOME=$XLE_CACHE/torch
export XDG_CACHE_HOME=$XLE_CACHE/xdg
export APPTAINER_CACHEDIR=$XLE_CACHE/apptainer

# Accept the Isaac Sim EULA without a prompt
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y

mkdir -p "$XLE_IMAGES" "$XLE_THIRD_PARTY" "$XLE_ASSETS" "$XLE_DATA" "$XLE_LOGS" "$XLE_PYDEPS" "$XLE_CONTAINER_HOME" \
         "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$APPTAINER_CACHEDIR" \
         "$XLE_CACHE/kit/cache" "$XLE_CACHE/kit/data" "$XLE_CACHE/kit/logs"
