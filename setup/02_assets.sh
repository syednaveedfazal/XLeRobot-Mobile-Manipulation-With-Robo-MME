#!/bin/bash
# 02_assets.sh - download the LeIsaac scenes and robots we can get without an account.
#
# Runs on a login node: these are plain downloads (about 130 MB) with no compute.
# Files land in $XLE_ASSETS, which LeIsaac finds through LEISAAC_ASSETS_ROOT.
#
# The Lightwheel Loft is NOT fetched here. It is only offered through
# lightwheel.ai, so download it by hand and unpack it to:
#   $XLE_ASSETS/scenes/lightwheel_loft/LW_Loft.usd
#
# Run from the repo root:  bash setup/02_assets.sh

set -euo pipefail
source env.local.sh

HF=https://huggingface.co/LightwheelAI/leisaac_env/resolve/main/assets
GH=https://github.com/LightwheelAI/leisaac/releases/download/v0.1.0

mkdir -p "$XLE_ASSETS/robots" "$XLE_ASSETS/scenes"
curl -fL --retry 3 -o "$XLE_ASSETS/robots/lekiwi.usd"         "$HF/robots/lekiwi.usd"
curl -fL --retry 3 -o "$XLE_ASSETS/robots/so101_follower.usd" "$HF/robots/so101_follower.usd"

curl -fL --retry 3 -o "$XLE_ASSETS/scenes/kitchen_with_orange.zip" "$GH/kitchen_with_orange.zip"
unzip -q -o "$XLE_ASSETS/scenes/kitchen_with_orange.zip" -d "$XLE_ASSETS/scenes"
rm "$XLE_ASSETS/scenes/kitchen_with_orange.zip"

ls -lh "$XLE_ASSETS/robots" "$XLE_ASSETS/scenes"
