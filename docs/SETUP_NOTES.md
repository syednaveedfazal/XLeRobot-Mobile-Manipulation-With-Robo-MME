# Setup notes

Cluster facts found, the chosen install route, versions and deviations.
Findings dated 2026-11-06.

## Phase 0: cluster facts

### Login node (login02)

| Item | Finding |
| --- | --- |
| OS | AlmaLinux 9.6 |
| glibc | 2.34 |
| Kernel | 5.14.0-570.35.1.el9_6 |
| Apptainer | 1.4.2 at `/usr/bin/apptainer`; `singularity` also present |
| Conda | Not on PATH; `module load Miniforge3` per the wiki |
| System compilers/tools | gcc 11.5, Python 3.9, git 2.47, unzip |
| Internet | `pypi.nvidia.com` reachable |

### Slurm

Slurm account: `ag_ifi_blum`, QOS `normal`.

| Partition | Time limit | GPUs per node | Nodes |
| --- | --- | --- | --- |
| `mlgpu_devel` | 1 h | 8 × A40 | 24 |
| `mlgpu_short` | 8 h | 8 × A40 | 24 |
| `mlgpu_medium` | 1 day | 8 × A40 | 24 |
| `mlgpu_long` | 7 days | 8 × A40 | 24 |
| `sgpu_*` | same tiers | 4 × A100 | 32 |

The `sgpu_*` A100 nodes have no RT cores, so Isaac Sim cannot use them.

The A40 partitions were busy on 2026-11-06, with about 146 jobs pending across
`mlgpu_devel` and `mlgpu_short`. Waits vary: one short job
sat for over 10 minutes without starting, and a later one started in under a minute.

### Storage

| Location | Details |
| --- | --- |
| `$HOME` | 100 GB quota, persistent; 2.3 GB used |
| Workspaces on `scratch` (default) | HDD, about 5 PB; no size limit |
| Workspaces on `mlnvme` (`-F mlnvme`) | SSD, 307 TB; for heavy file I/O |

`/lustre/scratch` itself is not writable. Lustre space comes from workspaces,
created with `ws_allocate <name> <days>`, which gives
`/lustre/scratch/data/<user>-<name>`.

- A workspace lasts up to 90 days and can be extended 3 times, 360 days in total.
- **Expired workspaces are deleted after a further 21 days.** Anything worth
  keeping must be rebuildable from this repo or copied elsewhere.
- Email reminders only work with an `@uni-bonn.de` address.
- `/lustre` was 94% full on 2026-11-06.
- The user currently has no workspaces.

### Cluster rules that affect this project

- Anaconda and Miniconda are not allowed, nor are the Anaconda default channels.
  Use Miniforge and conda-forge. The Route A command in the setup instructions
  uses `-c "nvidia/label/cuda-12.8.1"`, which is a separate channel.
- `gpu.marvin.hpc.uni-bonn.de` (`login03`) uses the same AMD software stack as the
  GPU nodes. The wiki says that submitting GPU jobs from the Intel login nodes
  can interfere with module stacks, so GPU work should be submitted from there.

### Isaac Sim availability

| Check | Finding |
| --- | --- |
| `isaacsim` 5.1.0 pip wheels | Tagged `manylinux_2_35_x86_64`, so pip refuses them on glibc 2.34 |
| NGC `nvidia/isaac-sim` | Anonymous pull token works; tag `5.1.0` exists |
| NGC `nvidia/isaac-lab` | Anonymous pull token works; tags `2.3.0`, `2.3.1`, `2.3.2` exist |

### A40 node (mlgpu008, job 28730078)

| Item | Finding |
| --- | --- |
| GPU | NVIDIA A40, 46068 MiB |
| Driver | 580.65.06 (CUDA 13.0), exactly the Isaac Sim 5.1 minimum |
| OS / glibc | AlmaLinux 9.6 / 2.34, same as the login node |
| NVENC | Encoder reported, `libnvidia-encode` present |
| Graphics libraries | `libGLX_nvidia`, `libEGL_nvidia`, `libnvidia-glcore`, `libvulkan.so.1` present |
| Vulkan ICD | `/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json` |
| `vulkaninfo` | Not installed on the node |
| Internet | `pypi.nvidia.com` and `github.com` return 200; `nvcr.io/v2/` returns 401, the normal reply without a token |
| Local disk | `/tmp` on NVMe, 1.7 TB free; `TMPDIR=/tmp` |
| RAM | 500 GB per node, shared between jobs |

Only one A40 node was probed. The driver on the other nodes is assumed to match.

### Apptainer `--nv` and Vulkan

`/etc/apptainer/nvliblist.conf` lists the graphics libraries Isaac Sim needs,
including `libnvidia-rtcore` and `libnvoptix`. The host's ICD file has a different
name from the one `--nv` expects, but the image ships its own
`/etc/vulkan/icd.d/nvidia_icd.json`, which points at `libGLX_nvidia.so.0`. So no
extra bind should be needed. The smoke test confirms this.

## Chosen route

**Route B: Apptainer, Isaac Sim 5.1 + Isaac Lab v2.3.0.**

- glibc 2.34 is below 2.35, so the native pip route (Route A) cannot install.
- Driver 580.65.06 meets the Isaac Sim 5.1 minimum, so no downgrade is needed.
- The A40 nodes are available, and `isaac-lab:2.3.0` pulled without an API key.

One image is enough: `isaac-lab:2.3.0` (7.8 GB `.sif`) contains Ubuntu 24.04,
glibc 2.39, Isaac Sim 5.1.0, Isaac Lab 2.3.0, Python 3.11.13, torch 2.7.0+cu128 and
numpy 1.26.0. LeIsaac pins Isaac Lab v2.3.0, so this matches.

## Deviations

- Added `slurm/phase0_probe.sbatch`, which is not in the planned layout, so the
  compute-node checks can run as a queued batch job instead of an interactive
  `srun`.
- Workspace `xle_sim` on `scratch` (`/lustre/scratch/data/s93nsyed_hpc-xle_sim`),
  expires 2027-02-04. No email reminder is set yet.
- Pulled only `isaac-lab:2.3.0`, not a separate `isaac-sim` image.
- Python packages go to `$XLE_PYDEPS` with `pip --user` and `PYTHONUSERBASE`, because
  the image is read-only. pip uses the legacy resolver because the strict one
  rejects `imageio[ffmpeg]` against Isaac's bundled imageio.
- **Lightwheel Loft dropped by the user.** It is only offered through lightwheel.ai.
  The Phase 1 smoke test uses `LeIsaac-SO101-PickOrange-v0` in the Kitchen.
  Phases 2 and 3 use a custom scene built from primitives (floor, table, objects,
  bins, cups, occluders), approved by the user.
- **Robot asset:** LeIsaac has no XLeRobot or bi-arm LeKiwi config. Use the USD in
  `xlerobot.zip`, which already has the virtual joints `root_x_axis_joint`,
  `root_y_axis_joint` and `root_z_rotation_joint`. Fallback: convert the URDF from the
  same zip. `simulation/xlerobot/xlerobot.urdf` is not used: it has a floating base and
  two wheels, while the real robot has three omni wheels.

## Phase 1 results

Job 28731031 on an A40:

- `list_envs.py` showed 15 environments, including `LeIsaac-SO101-PickOrange-v0` and
  `LeIsaac-LeKiwi-CleanupTrash-v0`.
- Smoke test on PickOrange: 300 steps at 21.6 steps/s. `front` mean 112.3, `wrist`
  mean 160.5, so neither is black. Images viewed and correct.
- Vulkan worked in the container with no extra bind.

Harmless log messages: `Failed to create NGX context` (DLSS is unavailable), and
PhysX failing on one Kitchen mesh, `outlet_room`.

`pip check` mismatches left as they are: for LeRobot, `deepdiff` 9.1.0, `packaging` 23.0
and `wandb` 0.22.2; `rerun-sdk` wants numpy 2. Revisit if recording fails.

## Open issues

- ~~Wrist cameras swapped~~: resolved. In `xlerobot.zip` the `_2` chain sits at +y (left)
  and carries `Left_Arm_Camera`. The confusing `_L`/`_R` names are only in the repo URDF.
- Joint limits differ between the USD and the repo URDF. Check against the real robot.
- ~~Shipped USD visuals~~: resolved. 69 meshes and 25 collision prims; it renders completely.
- `tools/build_xlerobot_usd.py` is not needed, because the zip USD already has the virtual base
  joints. Replaced by `setup/04_xlerobot_asset.sh` (unzip). The URDF converter is the fallback.
- Phase 2 smoke test (job 28756079): PASS. Base error 0.03 cm / 0.07°, all joints reach
  their targets, upright, nothing non-finite. All 4 cameras checked by eye.
- The wrist camera rotation is ours, not the URDF's: the URDF frame gave a 90° rolled image.

## Sim-to-real gaps (Phase 2)

The robot model is `simulation/Isaac_sim/xlerobot.zip` from XLeRobot commit `b017b5e`,
unchanged. The repo's two official models disagree:

| Measurement | `xlerobot.zip` (used) | `simulation/xlerobot/xlerobot.urdf` |
| --- | --- | --- |
| Arm mount spacing | 0.266 m | 0.22 m |
| Arm mount height above base frame | 0.76 m (floor level) | 0.395 m (chassis frame) |
| Head camera height above arm mounts | about 0.43 m | about 0.41 m |
| Arm direction at zero pose | forward | sideways |

Not yet checked against the physical robot. Gaps added on top of the USD:

- Virtual base joints: perfect velocity tracking, with no wheel slip or acceleration limit.
- Arm gravity off, as LeIsaac does for LeKiwi.
- SO-101 gains from LeIsaac, not tuned.
- Self-collisions off.
- Camera FOV of 75° and 640×480 are placeholders.
- The joint-to-motor mapping (`_ARM_SIGN_OFFSET`, `_HEAD_LIMITS_DEG`) is estimated.
- `head_motor_1` is assumed to be pan.

Needed from the user: real arm spacing, mount height and head camera height; camera
models and resolutions; LeRobot calibration JSONs for the arms and head; which head
motor is pan.

### Real-robot hardware (from the user's BOM sheet, "Copy of XLeRobot EU BOM")

The sheet is a parts list with no assembly dimensions. What it settles:

- 17 × FeeTech STS3215-C018, 12 V, 30 kg·cm. All sim arm, gripper and head actuators now
  use 2.94 N·m and 4.7 rad/s; it was 10 N·m before.
- IKEA RÅSKOG cart and the XLeRobot v0.3 print set, which matches the zip model's 0.76 m arm mount height.
- 3 × 100 mm Nexus omni wheels: holonomic, so the virtual x/y/θ base is the right model.
- Wrist cameras: InnoMaker 720p UVC, about 115–130° → sim 120°, 640×360.
- Head: Intel RealSense D435 or D455, model not confirmed. Sim uses D435 colour, 69° and
  640×360, until the model is confirmed. The D455 colour FOV is about 90°.
- Soft FinRay TPU fingers: the sim has rigid jaws. A known gap.

The repo has no assembled CAD: the STEP files are single parts and the `.3mf` is a print
plate. Exact mount positions must be measured on the built robot.

## Phase 2 revision: official SO-101 arms

- XLeRobot's model (`xlerobot.zip`) has **SO-100** arm meshes (`Base`, `Rotation_Pitch`,
  `Wrist_Pitch_Roll`, ...). The user's robot has SO-101 arms (BOM: `Ender_Follower_SO101`).
- The arms are now the official `so101_new_calib_camera.urdf` from TheRobotStudio/SO-ARM100
  (commit `a758567`, files in `$XLE_THIRD_PARTY/so_arm100_urdf`), merged by
  `tools/build_xlerobot_urdf.py` and converted by `tools/build_xlerobot_usd.py`.
- Mount: `left/right_arm_base_link` at (−0.1286, ±0.133, 0.7624) m in `base_link`. Computed by
  lining up the pan axes (the base footprints are identical: 0.096 m wide, pan axis 0.022 m from
  centre) and the base bottom faces. Shoulder-lift height is 0.879 m in both the old and new
  models. Both arms have the same orientation, not mirrored, as two standard SO-101s do.
- Upper and lower arm lengths matched the SO-100 model exactly (0.116 / 0.135 m); the
  shoulder, wrist, limits and masses did not.
- Mass: the total is 12.00 kg, from the docs. Massless links (virtual base links, head camera
  frames) get 0.001 kg, because PhysX treats 0 as 1 kg.
- Workspace check: tool point 0.55–1.29 m high (docs ~0.5–1.25 m); side reach 0.33 m beyond
  the body (docs ~0.36 m beyond the cart edge); `base_link` box height 0.775 m (RÅSKOG 0.78 m).
- Wrist camera: at the official `wrist_camera_link`. The lens looks along +z, and the barrel
  ends at 26.3 mm, so the camera sits at z = 27 mm. The roll puts the jaw tip at the image
  bottom; it is computed from the URDF. My first attempt (−z) was wrong: it looked back at
  the gripper.
- Arm conversion is now LeIsaac's SO-101 conversion, unchanged. The estimated sign/offset
  table is gone. The head mapping is still estimated.

## Phase 3 results

- `XLeRobot-Sandbox-v0`: 5 scripted episodes recorded (job 28759606), 661 frames each, HDF5.
- Conversion with LeIsaac's `isaaclab2lerobotv3.py`, unchanged, via `tools/convert_to_lerobot.py`
  (job 28761375): 5 episodes, 3280 frames (LeIsaac skips the first 5 of each), 30 fps, at
  `$XLE_DATA/lerobot/local/xle_sandbox_v0`. Keys are the real robot's 17, and cameras are `head`,
  `left_wrist` and `right_wrist` at 360×640. Value ranges match the commanded poses.
- Replay of episode 0: joints match the recording exactly (0.0 difference), and the red cube's end
  position matches to float precision.
- Deviations / fixes:
  - LeIsaac's StreamingRecorderManager only supports EXPORT_ALL; the converter's
    "successful only" filter does the selection. The record script marks a finished routine as success.
  - pyarrow pinned to 21.0.0 (setup/constraints.txt): pyarrow 26 needs NumPy 2, and the image
    has 1.26.
  - Kit hangs on shutdown after a Python error; the Isaac scripts now print the traceback and
    `os._exit(1)`. Batch scripts use `grep --line-buffered`.
- Known limitation: the scripted "reach" pose dips into the cart's own top basket (self-collisions
  are off), so it is a pipeline test, not a meaningful reach.

## Teleop (merged from xlerobo_setup)

- Branch `xlerobo_setup` merged locally (846f4a2). Its ManiSkill teleop page, server, relay and
  input mappings are reused for Isaac (`xle_leisaac/teleop/`, commit 7162f53).
- Its `teleop_web.py` imports `tyro`, which needs a newer `typing_extensions` than Isaac bundles;
  the Isaac adapter registers an empty `tyro` placeholder instead (tyro is only its CLI).
- Self-test: base 0.40 m forward, 0.40 m left, 120° turn as commanded; 17 steps/s with 5 cameras.
