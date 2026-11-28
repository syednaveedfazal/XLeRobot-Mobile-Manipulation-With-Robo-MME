# Learning notes

Plain-language explanations, updated after each phase.

## Phase 0: what we learned about the cluster

**Login nodes and compute nodes.** You land on a login node when you connect.
It has no GPU and is shared, so it is only for editing, downloading and
submitting jobs. Real work runs on compute nodes, which Slurm hands out.

**Slurm partitions.** A partition is a queue for a group of nodes. On Marvin the
name says what you get: `mlgpu_` means A40 nodes, and `_devel`, `_short`,
`_medium`, `_long` set the time limit (1 h, 8 h, 1 day, 7 days). Shorter limits
are for testing.

**Why only the A40 nodes.** Isaac Sim renders with ray tracing, which needs RT
cores. The A40 has them and the A100 does not, even though the A100 is the
stronger compute GPU.

**Why a container.** Linux programs are built against a minimum version of glibc,
the core C library. Isaac Sim's pip packages need 2.35 and Marvin has 2.34, and
you cannot upgrade glibc without admin rights. A container carries its own
userspace, including a newer glibc. The GPU driver still comes from the host,
which is why the host driver version matters. Apptainer is the container tool
used on clusters because it needs no root access.

**Workspaces.** Big files go in a workspace on Lustre, not in `$HOME` (100 GB
limit). Workspaces expire, so everything in this repo is designed to be rebuilt
by script.

### Commands to try by hand

```bash
sinfo -s                      # partitions and how busy they are
squeue -u $USER               # your jobs
sacct -u $USER -S today -X    # your jobs from today, including finished ones
ws_list                       # your workspaces

# Look at an A40 node (queues, then writes the report to your home directory)
sbatch --output=$HOME/xle_probe_%j.out slurm/phase0_probe.sbatch
```

## Phase 1: what got installed

**The three layers.** Isaac Sim is NVIDIA's simulator: physics plus rendering.
Isaac Lab sits on top and adds a robot-learning framework, with environments made
of a scene, observations, actions and resets. LeIsaac sits on top of that and adds
SO-101 and LeKiwi robots, teleoperation, and recording to LeRobot datasets.

**Headless** means running with no window. The GPU still renders the camera
images, but they go to memory or files instead of a screen.

**How our setup fits together.** One read-only container image holds Isaac Sim,
Isaac Lab, Python and torch. Everything that must be written lives in the workspace:

| Need | Where | How |
| --- | --- | --- |
| Extra Python packages | `$XLE_PYDEPS` | `pip --user` with `PYTHONUSERBASE` |
| Kit caches | `$XLE_CACHE/kit` | binds in `slurm/in_container.sh` |
| Container home folder | `$XLE_CONTAINER_HOME` | `--home` |
| Scenes and robots | `$XLE_ASSETS` | `LEISAAC_ASSETS_ROOT` |

Read `slurm/in_container.sh` first: it is the single place where all of this is set.

**Why Isaac scripts start the way they do.** `AppLauncher` must run before any other
`isaaclab` or `leisaac` import, because those modules need the simulator running.
That is also why importing `leisaac` outside a simulation prints an `omni.physics`
error, which is harmless.

### Commands to try by hand

```bash
# Interactive shell inside the container on an A40 (run from the repo root)
bash slurm/sim_interactive.sh
/isaac-sim/python.sh tools/smoke_leisaac.py --enable_cameras

# The same thing as a batch job
sbatch --output=$HOME/xle_smoke_%j.out slurm/sim_job.sbatch \
    /isaac-sim/python.sh tools/smoke_leisaac.py --enable_cameras
```

Results land in `$XLE_LOGS/smoke/`.


## Phase 2: adding XLeRobot

**How a USD articulation works.** A robot in USD is a tree of rigid bodies (links)
joined by joints. One prim carries `ArticulationRootAPI`, which tells PhysX to solve
the whole tree together, in reduced coordinates. That is far more stable than
simulating each link separately with constraints. `tools/inspect_usd.py` prints this
tree for any USD.

**Why the base uses virtual joints.** Simulating omni wheels on a floor means
contacts, friction and slipping rollers: slow, and often unstable. Instead the
base hangs from three invisible joints, `root_x`, `root_y` (sliding) and `root_z_rotation`
(turning), attached to a fixed root. Moving those joints moves the robot. The
`xlerobot.zip` model already has them, the same idea as LeKiwi's `base_x/base_y/base_theta`.
The x and y joints slide along world axes, so a "forward" command has to be rotated
by the robot's heading first (`body_to_joint_velocity`).

**`ArticulationCfg` and actuator gains.** `XLEROBOT_CFG` in
`xle_leisaac/robots/xlerobot.py` says which USD to load, its start pose, and how
each joint is driven. An implicit actuator is a spring and damper:
`torque = stiffness × (target − position) + damping × (target_speed − speed)`.

- Arms (stiffness 17.8, damping 0.6): copied from LeIsaac's SO-101.
- Base (stiffness 0, damping 100000): only the speed term remains, so the joint
  becomes a velocity controller.

**The 17-D action.** The order matches the real robot's action keys:

| Index | Real key | Sim joint | Sim unit | Real unit |
| --- | --- | --- | --- | --- |
| 0–5 | `left_arm_{shoulder_pan…gripper}.pos` | `left_arm_shoulder_pan … left_arm_gripper` | rad | −100…100 (gripper 0…100) |
| 6–11 | `right_arm_{…}.pos` | `right_arm_shoulder_pan … right_arm_gripper` | rad | same |
| 12, 13 | `head_motor_1.pos`, `head_motor_2.pos` | `head_pan_joint`, `head_tilt_joint` | rad | −100…100 |
| 14, 15 | `x.vel`, `y.vel` | base forward / left speed | m/s | m/s |
| 16 | `theta.vel` | base turn speed | rad/s | deg/s |

`sim_to_lerobot_action()` and `lerobot_to_sim_action()` convert between the two.

### Commands to try by hand

```bash
bash setup/04_xlerobot_asset.sh   # unpack XLeRobot's model (body, cart, head)
python3 tools/build_xlerobot_urdf.py   # body + two official SO-101 arms -> one URDF
slurm/in_container.sh /isaac-sim/python.sh tools/build_xlerobot_usd.py   # URDF -> USD
sbatch --output=$HOME/xle_xsmoke_%j.out slurm/sim_job.sbatch \
    /isaac-sim/python.sh tools/smoke_xlerobot.py --enable_cameras
```

Frames and videos land in `$XLE_LOGS/smoke_xlerobot/`.

### Why the arms come from a different repo

XLeRobot's own simulation model has SO-100 arms, the older version. The real robot uses
SO-101 arms, and the two differ at the shoulder and wrist, and in joint limits and masses.
So `tools/build_xlerobot_urdf.py` keeps XLeRobot's body and swaps in the official SO-101
model from TheRobotStudio/SO-ARM100.

To place it, the script lines up the shoulder-pan axes and the bottom faces of the arm
bases, which have identical footprints. As a cross-check, the shoulder-lift axis lands at
0.879 m, exactly where the old model had it.

The official "new_calib" model is defined to match LeRobot's calibration, so a sim angle
of 0 means the same as a real calibrated reading of 0.

**URDF vs USD.** URDF is the robot-description format from ROS: links, joints, meshes, as
XML. Isaac Sim uses USD. Isaac Lab's `UrdfConverter` turns one into the other. We edit
the URDF, because it is easy to read and script, then convert.

## Phase 3: sandbox, recording, and datasets

**The sandbox environment** (`xle_leisaac/tasks/sandbox_env_cfg.py`) is a manager-based Isaac Lab
environment. Read it top to bottom: each part is one small, commented class.

| Part | What it does here |
| --- | --- |
| Scene | floor, light, robot, cameras, table, 3 cubes, cylinder, 2 bins, 2 cups, 2 panels, partition |
| Observations | `state` (the 17 values) plus `head`, `left_wrist`, `right_wrist` images |
| Actions | `joints` (14 absolute targets), then `base` (forward, left, turn) = 17 values |
| Events | on reset: put everything back, then shift and spin the objects randomly (seeded) |
| Terminations | only `time_out` after 60 s, so there is no success condition |

**How LeIsaac records (HDF5).** A *recorder manager* is attached to the environment. Every
step it stores the action, the observations and the full scene state (robot joints, object
poses). It also stores the starting state of each episode. When an episode ends, the episode is
written into one HDF5 file as `data/demo_0`, `data/demo_1`, ... HDF5 is a container format: a
file system inside one file. Raw images make it big: 5 episodes are 4.3 GB.

Because the starting state is saved, an episode can be **replayed**: restore the start, send
the same actions, and the sim repeats it exactly (`tools/replay_episode.py`).

**What a LeRobot dataset is.** The format the LeRobot library trains on (v3):

```
xle_sandbox_v0/
  meta/info.json        features (names, shapes), fps, robot type, counts
  meta/stats.json       min / max / mean / std of every feature (for normalisation)
  meta/tasks.parquet    the task text of each episode
  meta/episodes/...     where each episode starts and ends
  data/chunk-000/*.parquet       one row per frame: action, observation.state, indices
  videos/<camera>/chunk-000/*.mp4  camera streams, compressed (the 4.3 GB HDF5 became 120 MB)
```

Parquet is a column-based table format, like a compact spreadsheet. The converter turns the
simulator's units (radians, rad/s) into the real robot's: arm values −100…100, gripper
0…100, `x.vel`/`y.vel` in m/s, `theta.vel` in deg/s. So sim and real data look the same to
a policy.

**How to add your own task** (for example a memory task), using the sandbox as a template:

1. Copy `xle_leisaac/tasks/sandbox_env_cfg.py` to `xle_leisaac/tasks/my_task_env_cfg.py`.
2. Change the scene: add, remove or move objects in `SandboxSceneCfg`.
3. Add a **success** condition: a function `def task_done(env) -> torch.Tensor` (one bool per
   env) and `success = DoneTerm(func=task_done)` in the terminations. LeIsaac's tools keep only
   successful episodes.
4. Add reset events for anything that should vary between episodes.
5. Register it in `xle_leisaac/__init__.py` with a new id, e.g. `XLeRobot-MyTask-v0`.
6. Record: drive it with the browser teleop, or write a script like `tools/record_scripted.py`.
   Then convert with `tools/convert_to_lerobot.py` (change the task name and repo id).

LeIsaac's own guide to the same idea: `$XLE_THIRD_PARTY/leisaac/docs/docs/docs/tutorials/custom_task.md`.

### Commands to try by hand

```bash
sbatch --output=$HOME/xle_data_%j.out --time=00:55:00 slurm/sim_job.sbatch bash -c '
  /isaac-sim/python.sh tools/record_scripted.py --enable_cameras --episodes 5 &&
  /isaac-sim/python.sh tools/convert_to_lerobot.py --overwrite &&
  /isaac-sim/python.sh tools/check_dataset.py'
```
