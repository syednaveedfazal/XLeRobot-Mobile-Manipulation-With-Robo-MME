# XLeRobot-Mobile-Manipulation-With-Robo-MME

Memory-dependent mobile manipulation on [XLeRobot](https://github.com/Vector-Wangel/XLeRobot)
(sim + real), evaluated with π0.5. Uni Bonn, Robot Perception and Learning Lab.

**Current state:** XLeRobot runs in ManiSkill (ReplicaCAD apartment) on the Marvin cluster and can be
teleoperated from your browser with the keyboard or an Xbox (or any standard) gamepad.

## Layout

```
setup/install.sh, env.sh, requirements.txt   one-time install / activate the environment
xlerobot_sim/__init__.py                     registers agent "xlerobot" + env "XLeRobot-ReplicaCAD-v1"
xlerobot_sim/agent.py                        upstream agent + fixes so it matches the real robot (see below)
xlerobot_sim/layout.py                       real-robot action/state order (17 keys) and sim joint mapping
xlerobot_sim/teleop_controller.py            keyboard / gamepad -> command -> 17-dim action
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
| View (⧉) + D-pad left/right | base slide left/right |
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
| Q / E | base slide left/right |
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
               control_mode="pd_joint_pos_real_layout", render_mode="rgb_array")
obs["agent"]["state_real_layout"]   # (num_envs, 17), same order as the action
```

## How the sim model matches the real robot

`xlerobot_sim/agent.py` subclasses the vendored upstream agent; the files in `third_party/` stay
unmodified. Compared with upstream:

| | upstream ManiSkill model | now | real XLeRobot |
|---|---|---|---|
| mass | 79.3 kg (base_link 70.1 kg) | 12.3 kg | ~12 kg |
| cameras | head 256x256, wrists 128x128 | all 640x480; head 69° / wrists 120° horizontal FOV | 640x480 |
| action (`pd_joint_pos_real_layout`) | 16-dim, base first, right arm first, deltas | 17-dim in the real key order | 17 keys |
| base | forward + turn only, up to 1 m/s | x, y, yaw velocity in the robot frame, clamped to 0.3 m/s, 90°/s | omni wheels, fastest preset 0.3 m/s, 90°/s |

Action / state order (`xlerobot_sim.layout.ACTION_KEYS`): left arm `shoulder_pan, shoulder_lift,
elbow_flex, wrist_flex, wrist_roll, gripper` (0-5), right arm (6-11), head pan, tilt (12-13),
`x.vel, y.vel, theta.vel` (14-16).

The upstream 16-dim modes (e.g. `pd_joint_delta_pos_dual_arm`) are still available and unchanged.

### Things to look out for

Still open. Check these before you mix sim data with real-robot data or train on it.

- **Arm zero points don't match the real arm.** The sim arm is the older SO-100-style URDF (shoulder
  lift -0.1..3.45 rad, elbow -0.2..3.14 rad; the rest pose has both near 3.14). The real SO-101 is
  calibrated around zero (lift ±100°, elbow ±96.8°, wrist roll -157°..163°). The same number means a
  different arm pose in sim and on the robot. **To do:** swap in the official SO-101 arm (as the
  Isaac build does) or write a per-joint offset/sign table from the SO-101 URDF. Don't guess the
  table; a wrong one silently corrupts every converted episode.
- **Units differ.** Sim joints are radians; the real robot reports -100..100 (gripper 0..100), or
  degrees with `use_degrees=True`. `theta.vel` is rad/s in sim and deg/s on the real robot. `x.vel`
  and `y.vel` are m/s in both.
- **Head motor order is a guess.** `head_motor_1` = pan and `head_motor_2` = tilt is assumed in
  `xlerobot_sim/layout.py`. **To do:** move each head motor on the real robot and fix the mapping if
  it's the other way round.
- **Isaac model not updated.** The Isaac model (`xle_leisaac`, not in this repo) still has an
  unclamped base (velocity limit 100) and 640x360 cameras. **To do:** clamp the base to 0.3 m/s and
  90°/s, and pick one camera size (640x480 like the real cameras) for both sims.
- **Wrist camera images are rotated about 90°.** This comes from upstream's camera mount on the
  gripper (wrist roll rests at 1.57 rad). It was easy to miss at the old 128x128. **To do:** check it
  against a real wrist-camera frame and rotate the sim mount or the image to match.
- **Teleop feels different from before.** Teleop now sends absolute joint targets (at most 0.1 rad
  change per step) instead of P-controlled deltas. The gripper opens wider (1.7 rad, the jaw limit)
  and the base moves at the real "slow" preset. Expect slightly different arm and gripper behaviour
  from earlier sessions.

## Notes

- **GPU:** the `mlgpu_*` partitions have A40s, which support graphics and ray tracing. The `sgpu_*`
  A100s can render the default shader but have no RT cores. Physics runs on the CPU for a single env.
- **Versions:** mani-skill 3.0.0b21, sapien 3.0.3 and torch 2.15 (CUDA 13). Marvin's driver is 580,
  which supports CUDA 13.0.
- **Upstream pygame demos** (`third_party/xlerobot_maniskill/examples/`) open a local window and the
  SAPIEN viewer. They need a real desktop session, so they don't run in a plain Slurm job.
- **Changes from upstream:** the third-person camera follows from behind instead of facing the robot.
  Teleop sends absolute joint targets (rate-limited to 0.1 rad per step) and base velocities instead of
  P-controlled deltas; base speeds are the real robot's "slow" preset (0.1 m/s, 30°/s).
  Key X keeps the base heading instead of turning the robot back to 0. Grippers start open, and keyboard
  V/B toggles them closed.
