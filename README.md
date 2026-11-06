# XLeRobot-Mobile-Manipulation-With-Robo-MME

Memory-dependent mobile manipulation on [XLeRobot](https://github.com/Vector-Wangel/XLeRobot)
(sim + real), evaluated with π0.5. Uni Bonn, Robot Perception and Learning Lab.

**Current state:** XLeRobot runs in ManiSkill (ReplicaCAD apartment) on the Marvin cluster and can be
teleoperated from your browser with the keyboard or an Xbox (or any standard) gamepad.

## Layout

```
setup/install.sh, env.sh, requirements.txt   one-time install / activate the environment
xlerobot_sim/__init__.py                     registers agent "xlerobot" + env "XLeRobot-ReplicaCAD-v1"
xlerobot_sim/teleop_controller.py            keyboard / gamepad -> command -> 16-dim action
xlerobot_sim/teleop_web.py                   headless sim + browser teleop server
scripts/start_teleop.sh, teleop.sbatch       launch teleop as a Slurm job
scripts/relay.py                             login-node relay so VS Code can forward the page
scripts/smoke_test.py                        headless check: loads, renders, moves the right way
third_party/xlerobot_maniskill/              upstream XLeRobot ManiSkill files (URDF, meshes, agent), pinned
```

Nothing is copied into `site-packages/mani_skill`. The upstream guide asks you to copy files there and
edit two ManiSkill files by hand. `import xlerobot_sim` does the same work at import time.

## Setup on Marvin (once)

Run this on the login node, which has internet access:

```bash
bash setup/install.sh
```

The script:
- allocates the workspace `xlerobot` on `/lustre/scratch` (90 days)
- creates a Python 3.11 conda env inside it (`$WS/envs/xlerobot`)
- downloads ReplicaCAD into `$WS/maniskill_data`

Home stays small. Every new shell needs:

```bash
source setup/env.sh
```

> **Workspaces expire.** Check with `ws_list`. Extend with `ws_extend xlerobot 90` (3 extensions allowed).
> If it expires, the env and assets are deleted. Re-run `setup/install.sh` to rebuild them.

## Teleop in simulation

```bash
bash scripts/start_teleop.sh                        # 1 h on mlgpu_devel (A40)
PARTITION=mlgpu_short TIME=04:00:00 bash scripts/start_teleop.sh   # longer session
```

The script submits a GPU job and waits until the scene has loaded (usually under a minute). It then
starts a small relay on the login node (`scripts/relay.py`, localhost only) and prints a URL:

```
 Open in your laptop's browser:   http://localhost:22874/?token=...
```

1. Run the script from the **VS Code terminal**. VS Code forwards the printed port from the login node
   to your laptop. If it doesn't show up in the **Ports** tab, add the port number there yourself.
   The port is `20000 + your uid % 20000` (22874 for `s24imoha_hpc`). If it's taken, the next free one is used.
2. Open the URL. Click the page so it captures the keyboard. The robot stops when the page loses
   focus, when you press Esc, or when the browser stops sending input for 1 s.
3. When you're done, run `scancel <job>`. The relay stops on its own when the job ends.

Run the script from the terminal on the login node, not from your laptop. Without VS Code, use the
`ssh -N -L ...` command it prints in a terminal **on your laptop**.

The page shows a chase camera, the head camera and both wrist cameras. It runs at about 30 Hz.
Keyboard and gamepad can be used at the same time. **Arm 1 is the robot's right arm, arm 2 its left.**

### Gamepad (Xbox)

Connect the controller **to your laptop** (USB or Bluetooth). Open the page in Chrome or Edge, then press
any button on the controller. The header should say `gamepad: Xbox ...`. This uses the browser's Gamepad
API, so nothing is installed on the cluster. Any controller the browser reports with the `standard`
layout works the same way, e.g. PlayStation or Switch Pro.

| Gamepad | Action |
|---|---|
| D-pad | base forward/back, turn left/right |
| Right stick | arm 1 (right) end effector up-down / forward-back |
| RB + right stick | arm 1 wrist pitch / roll |
| R3 (click) + right stick | arm 1 shoulder rotation |
| RT | close gripper 1 (analog: release to open) |
| Left side (stick, LB, L3, LT) | the same for arm 2 (left) |
| X / B, Y / A | head pan left/right, tilt up/down |
| Menu (≡) | reset arm targets |

The layout follows upstream's `demo_ctrl_action_ee_xbox.py`, with two fixes: each trigger now closes the
gripper on its own side, and the base is driven with the D-pad.

**Joy-Con (planned primary device):** this isn't implemented yet. Untested Joy-Con input is safer to
build once the hardware arrives. When it does, pair the Joy-Cons with your laptop, open the page and tick
**raw gamepad**. The panel shows the layout and button numbers the browser reports, and that's what
the Joy-Con mapping will be built from.

### Keyboard

| Keys | Action |
|---|---|
| W / S, A / D | base forward/back, turn left/right |
| 8 / U, 9 / I | arm 1 end-effector up/down, forward/back |
| 7 / Y, 0 / O, - / P | arm 1 shoulder rotation, wrist pitch, wrist roll |
| J / M, K / , | arm 2 end-effector up/down, forward/back |
| H / N, L / ., ; / / | arm 2 shoulder rotation, wrist pitch, wrist roll |
| V / B | toggle gripper 1 / 2 |
| R / T, F / G | head pan, head tilt |
| X | reset arm targets |

Options: `bash scripts/start_teleop.sh --shader rt-fast` (ray traced, slower), `--deadzone 0.2` (gamepad). Run
`python -m xlerobot_sim.teleop_web --help` for the full list.

### Check without a browser

```bash
sbatch -p mlgpu_devel --gres=gpu:1 -c 8 --mem=32G -t 00:20:00 -o logs/smoke-%j.out \
       --wrap "source setup/env.sh && python scripts/smoke_test.py"
# -> "SMOKE TEST PASSED", images in outputs/
```

### Using the env in your own code

```python
import gymnasium as gym, xlerobot_sim
env = gym.make("XLeRobot-ReplicaCAD-v1", obs_mode="rgb",
               control_mode="pd_joint_delta_pos_dual_arm", render_mode="rgb_array")
```

## Notes

- **GPU:** the `mlgpu_*` partitions have A40s, which support graphics and ray tracing. The `sgpu_*`
  A100s can render the default shader but have no RT cores. Physics runs on the CPU for a single env.
- **Versions:** mani-skill 3.0.0b21, sapien 3.0.3 and torch 2.15 (CUDA 13). Marvin's driver is 580,
  which supports CUDA 13.0.
- **Upstream pygame demos** (`third_party/xlerobot_maniskill/examples/`) open a local window and the
  SAPIEN viewer. They need a real desktop session, so they don't run in a plain Slurm job.
- **Changes from upstream:** the third-person camera follows from behind instead of facing the robot.
  Key X keeps the base heading instead of turning the robot back to 0. Grippers start open, and keyboard
  V/B toggles them closed.
