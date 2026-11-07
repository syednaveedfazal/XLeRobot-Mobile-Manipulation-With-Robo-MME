#!/bin/bash
# 04_xlerobot_asset.sh - unpack the XLeRobot USD from the XLeRobot repo.
#
# The zip already contains a sim-ready robot with the virtual base joints, so
# nothing needs converting. The USD ends up at:
#   $XLE_ASSETS/robots/xlerobot/xlerobot/xlerobot/xlerobot.usd
#
# Run from the repo root:  bash setup/04_xlerobot_asset.sh

set -euo pipefail
source env.local.sh
mkdir -p "$XLE_ASSETS/robots/xlerobot"
unzip -q -o "$XLE_THIRD_PARTY/XLeRobot/simulation/Isaac_sim/xlerobot.zip" -d "$XLE_ASSETS/robots/xlerobot"
ls -l "$XLE_ASSETS/robots/xlerobot/xlerobot/xlerobot/xlerobot.usd"
