"""train_policy.py - `lerobot-train` with our policies (pi05_mem) registered.

Takes exactly lerobot-train's arguments. Runs inside the container with the pi0.5
packages on the path (XLE_WITH_PI=1), e.g. the no-memory baseline and FrameSamp+Modul:

  sbatch slurm/train_pi05_mem.sbatch none
  sbatch slurm/train_pi05_mem.sbatch framesamp_modul
"""

import xle_policies  # noqa: F401  (registers pi05_mem before LeRobot parses the CLI)
from lerobot.scripts.lerobot_train import main

if __name__ == "__main__":
    main()
