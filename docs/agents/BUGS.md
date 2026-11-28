# BUGS (sim-debugger and data-auditor)

Format per bug: symptom, how to reproduce, cause, fix, how it was verified.

## Open

- **B1 Block too small to see.** Symptom: trained policies drive to the wrong station. Reproduce:
  render a dataset head frame at pi0.5's input size (640x360 letterboxed into 224x224 = 224x126).
  Cause: the 2x2x4 cm block covers 2-3 px. Fix (in progress): 6 cm box with the 2 cm handle.
  Verified: not yet. D1 audit: at the first forward-drive frame (57, same for all 112 demos) the block is 5.2x7.0 / 3.8x4.9 / 2.8x4.2 px (w x h, median, station 0/1/2) at 224 px; 0/112 reach 10x10. At the moment the stop is decided it is 9x11 px, but only 2-8 % reach 10x10.
- **B2 Expert commits before the block is visible.** Symptom: demos drive to the right station from
  simulator state. Cause: `put_it_back_expert.py` read `env.put_it_back_station` from the first step.
  Fix (in progress): look first, drive slowly along the row until the box is >= 10x10 px at 224 px.
  Verified: not yet. D1 audit: the commit frame (first forward drive) is 57 in all 112 demos for every station. The block is in the head image then (tiny, see B1) but is out of view in frames ~70-150 (station 1 reappears at ~160; station 2 not visible up to frame 190) while the expert keeps driving to the station it read from the sim.
- **B3 Episode length reveals the hidden station (timing shortcut).** Found by data-auditor on D1 (2026-11-24).
  Symptom: length is 1545 +/- 4 / 1726 +/- 4 / 1905 +/- 5 frames for station 0/1/2 (ranges do not overlap);
  the turn back after the table starts at frame 1193 / 1314 / 1433 (+/- 4-5), i.e. at the memory decision point
  the elapsed time alone gives the station. A memory with time stamps (FrameSamp/Modul) can cheat by counting.
  Reproduce: `$XLE_WORK/scratch/a3.py` (recovers station from integrated x.vel, writes events.csv).
  Cause: the expert drives at a fixed 0.25 m/s, so the detour time grows 0.5 m / 0.25 m/s = 2 s per station, and
  every other phase has a fixed duration. Fix (proposal, D2): randomise the timing so that time carries no
  information, e.g. random pause (0-6 s) at the table and random wait after the grasp, drawn independently of the
  station; or a random cruise speed 0.15-0.25 m/s per episode plus random pauses; then re-check that the
  decision-frame distributions per station overlap (rank test p > 0.1). Also hold the number of frames fed to
  the memory fixed, or feed it frame order only. Fix applied for D2 (2026-11-24, not yet verified): random
  cruise speed 0.15-0.30 m/s per episode, random waits at the start (1-3 s), after the grasp (0-2 s) and at
  the table (1-9 s), all independent of the station (`put_it_back_expert.py`); per-episode station and timing
  saved in `<chunk>_episodes.json` and `meta/xle_episodes.json` of the dataset. Verify: rank test on D2.
- **B4 The expert's visibility check used a stale head-camera pose.** Found 2026-11-24 (box test, job 29353206).
  Symptom: "cube never clearly seen" for stations 0 and 2 although the head video shows the 6 cm box clearly
  (~12 px at station 0). Cause: `cube_pixels()` used the camera's stored pose (`data.pos_w`,
  `quat_w_ros`). Isaac Lab reads it from the USD stage, which is not updated for robot links during
  simulation, so it stays at the start pose (head looking straight ahead) even with
  `update_latest_camera_pose=True` (tried in job 29353270: no change). Fix: compute the camera pose from the
  physics pose of `head_tilt_link` plus the camera's mounting offset. Verified (job 29353285,
  `$XLE_LOGS/vis_debug/*.png`): the projected corners land on the box in the image; station 0 box
  13.6x18.6 px at 224 px, station 2 7-8x10 px growing as the robot approaches. Note for everyone: never
  use a camera's `data.pos_w`/`quat_w_*` for robot-mounted cameras in this repo.
- **B5 Recovery noise broke the grasp.** Found 2026-11-24 (job 29353206): 0/2 with noise. Cause: base noise was
  also applied in steps where the base has no goal (arm swing, reach), so nothing corrected the drift before the
  grasp. Fix: base noise only while the base controller steers to a goal (`expert.driving`), arm noise outside
  grasp/set-down. Verify: job 29353270.
- **B6 Eval: the first image of every episode is stale (eval only). FIX APPLIED 2026-11-24:** `self.rerender_on_reset = True` in `XLeRobotPutItBackEnvCfg.__post_init__` (covers eval, live viewer and recording). Verify on the next eval. Found by
  sim-debugger 2026-11-24 (train/eval parity check). Symptom: the first observation after a reset shows the
  scene from before the reset, while the 17 state values already show the reset pose. Episode 0: a render from
  before the reset events (cube at its default spot, the cart's blue tray and the floor grid missing); the wrist
  cameras differ from a still frame 60 steps later by 26.6/255 on average (about 50 % of pixels), from step 1
  on by < 1.2/255. Episodes 1+: the first image is the previous episode's last frame (0.36/255 from it, but
  100.8/255 and 98 % of pixels from the very next frame). Effect: the first plan of every eval episode
  (50 actions = 1.7 s) is made from a wrong picture, and the memory variant keeps it as its first memory frame.
  Training data never has this: the converter drops the first 5 recorded steps. Reproduce:
  `$XLE_WORK/scratch/parity/first_frames.py`, `stale_fix.py` (jobs 29353283, 29353281; pictures in
  `$XLE_WORK/scratch/parity/out/first_frames_*.png`, `stale_*.png`). Cause: Isaac Lab's `rerender_on_reset`
  is False by default, so the cameras keep their last render after a reset. Proposed fix: in
  `tools/eval_put_it_back.py`, after `env_cfg.recorders.dataset_export_mode = ...`, add
  `env_cfg.rerender_on_reset = True`; the same before `gym.make` in `xle_leisaac/teleop/isaac_teleop_web.py`.
  Verified in scratch (jobs 29353281, 29353288): with it the first image after an auto-reset differs from the
  next frame by 1.44/255 (normal frame-to-frame change 0.9/255) instead of 100.8/255, and episode 0's first
  frame is complete (wrists 2.1/255 from step 60 instead of 26.6/255). Probably not the main cause of 0/10:
  episode 0 fails too, and the first plan mostly moves the head and opens the gripper.
- **B7 Eval starts in a head pose the dataset never shows (eval only, low impact).** At eval step 0 the head is
  level: head_motor_1 0, head_motor_2 -31.2. The dataset's lowest head_motor_2 is -17.9 (q01 -4.05), because
  the converter drops the first 5 steps, in which the expert has already tilted the head. The state tokens in
  the prompt at step 0 match the dataset's frame 0 in 14 of 17 values (head_motor_1 255 vs 225, two others off
  by 1-2 bins), but the horizon view of the head camera is new to the policy. Lasts one plan. Proposal (not
  applied; the first is a data change): keep the first 5 steps in the converter, or let eval hold still for
  5 steps before the first plan.
- **B8 Image padding is -3 instead of -1 (training and eval alike; not a parity bug). FIX APPLIED for new runs 2026-11-24:** `PI05MemConfig.pad_black` (default False so T1 checkpoints behave as trained); `slurm/train_pi05_mem.sbatch` sets `--policy.pad_black=true`; `tools/build_memory_cache.py --pad_black 1` (default) and the cache meta records it, training refuses a mismatched cache. Needs a new memory cache and new training (planned anyway for D2). LeRobot 0.4.2 pi0.5
  (`_preprocess_images`) letterboxes the [0, 1] image with pad value -1, then maps x -> 2x - 1, so the black
  bars become -3. For our 640x360 cameras that is 98 of 224 rows (44 %) of every camera image, outside SigLIP's
  [-1, 1] input range, with the vision encoder frozen (`freeze_vision_encoder=true`). The memory features
  (`encode_memory_frames`) do the same. Reproduce: model input min is -3.000 for every camera (scratch
  `parity/ds_side.py`). Proposal: pad in [0, 1] space with 0 (black, -1 after mapping), e.g. override
  `_preprocess_images` in `xle_policies/pi05_mem`; needs retraining and a rebuilt memory cache.
- **B9 Training images are AV1-compressed, eval images are not (minor).** D1 stores the cameras as AV1
  (libsvtav1, crf 30, yuv420p); eval feeds the raw render. On 90 eval head frames the AV1 round trip changes
  pixels by 1.09/255 on average (PSNR 40.7 dB; 0.55/255 at 224 px), but small coloured parts fade: red handle
  pixels 233/66/55 -> 219/72/64, half of them no longer pass a strict red mask, sticker saturation 0.97 -> 0.91.
  Matters only for tiny objects. Proposal (recording change, not applied): a lower crf (e.g. 18-23) for D2.

### Checked and matching (parity check 2026-11-24, jobs 29353268, 29353281, 29353283, 29353288)

- Images: env gives RGB uint8 (1, 360, 640, 3) -> eval path -> float32 (1, 3, 360, 640) in [0, 1]; the
  dataloader gives the same shape, type and range. Image normalisation is IDENTITY in both; the model letterboxes
  both to 224x126 the same way. RGB order is right in both (yellow arm, blue/yellow stickers). A dataset frame
  sent through the eval path gives exactly the same 50 actions as through the training path (max diff 0.0).
- State: same function (`xlerobot_state` + `sim_to_lerobot_action`) and order in both; base speeds in the body
  frame (0 at the start in both), turn in deg/s. Task text identical. `sim_to_lerobot_action` and
  `lerobot_to_sim_action` are exact inverses inside the joint ranges (0 error real->sim->real on 1000 random
  vectors; 3.6e-8 rad sim->real->sim on the eval state); outside the ranges values are clamped, as on the robot.
- Actions: on the eval first frame the first plan looks like the demos (right gripper opens 12.6 -> 56, demo
  6 -> 57; head pan -15 -> -26, demo -19 -> -32; base speed ~0, demo 0), in the right sim units (gripper 0.067 rad).
- Block size (B1) re-measured: in dataset head frames at 224 px the block's coloured stickers cover 2-9 px at the
  start (episodes 7, 3, 0), 10-19 px while driving along the row, 0 px for several seconds while out of view,
  and ~55 px (about 8x8) only just before the grasp (`$XLE_WORK/scratch/parity/block_px.py`).
- Note: the left arm never moves, so its state quantiles are almost equal (shoulder pan q01..q99 is 0.016 units
  wide) and tiny noise there picks arbitrary state tokens. Eval and dataset agreed at step 0, but it is fragile.

- **B11 D2 side file `meta/xle_episodes.json` is in the wrong order (labels, not video/state data). Found by data-auditor 2026-11-26.**
  Symptom: the station stored under `dataset_episode` k is right for only 22 % of the 174 episodes (135 wrong). Cause: the
  converter writes the demos of a chunk in string order (episode 0, 1, 10, 11, 2, 3, ...) but `tools/merge_datasets.py` pairs the
  side file in recording (numeric) order. Verified: with within-chunk string order, the episode length equals `sim_seconds*30-5`
  (+-1 frame) in 174/174 episodes, and the station recovered from integrating x.vel (a3-style) agrees in 169/174 (57/57 noise off;
  the 5 misses are noisy episodes, where the clean actions do not integrate to the real position). Mapping file: `$XLE_WORK/scratch/d2_ds2meta.json`
  (dataset episode -> side-file index). Impact: none on the images/actions that training uses (no station label is read); any
  audit, eval pairing or per-station analysis that uses the file as stored is wrong. Fix (proposal, not applied): in
  `merge_datasets.py` sort each chunk's side entries with `key=lambda e: str(e["episode"])` before pairing (or have the converter
  keep numeric order), regenerate the file only, and keep the length check as an assertion.
- **B12 D2 still has a timing shortcut (B3 reduced, not removed). Found by data-auditor 2026-11-26.**
  Symptom: turn-back frame (memory decision point) is 1341 / 1514 / 1668 (mean; ranges 1127-1566 / 1237-1756 / 1448-2039) for
  station 0/1/2; Kruskal-Wallis p = 1e-22, Spearman rho 0.76. Episode length 1713 / 1966 / 2170 (p = 7e-24). A 1-D Gaussian classifier on
  the decision frame alone reaches 65 % (chance 33 %, permutation p < 0.003); on length 64 %. Also with noise off only (p = 3e-7).
  Cause: the random timing (start 1-3 s, grasp 0-2 s, table 1-9 s, speed 0.15-0.30) has a spread of about 2.5-3 s, but the station adds about 5.3 s
  per step: 2 x 0.5 m of extra driving, and the look-first approach (slow 0.12 m/s until the box is clearly visible, 0 / 0.4 / 4.0 s mean for
  station 0/1/2). Fix (proposal, for a later dataset): make the turn-back time independent of the station by construction, i.e. draw
  a target time T ~ U(60, 80) s per episode and set the table wait = T minus elapsed (after the detour), and pad the final rest
  step the same way so the length is also free of the station. Cheaper alternative: widen the table wait to 0-25 s (spread above
  about 7 s). Either needs a re-record; the already recorded D2 cannot be repaired. Mitigation for T2-mem on D2: report the
  memory result next to the time-only baseline (65 %), i.e. a memory run that only counts time cannot exceed about 65 % station accuracy.
  **Fix applied 2026-11-26 (for D3):** the table wait now lasts until a clock time WAIT_UNTIL ~ U(60, 75) s drawn per
  episode (put_it_back_expert.py), so the turn-back time no longer depends on the station; an episode that reaches the
  table after its drawn time is flagged "late". The expert logs every step's start time (side file: events, turn_back_at).
  Gates: the check job stops unless 5/6 noisy episodes succeed with none late; the merge stops if the turn-back time alone
  predicts the station above 45 % (same 1-D Gaussian classifier as the audit, 5-fold), or if more than 5 % are late.
  Episode limit 120 -> 150 s. Remaining by design: the length AFTER the turn-back still depends on the station (the
  drive back), which a policy cannot use for its decision.
- **B13 D3: station still shows in the timing of events before the decision (low, not blocking). Found by data-auditor 2026-11-26.**
  The decision time itself is clean (B12 fixed: 29 % vs chance 33 %). But the first pickup takes longer for farther stations:
  first grasp at 11.9 / 14.4 / 18.2 s, arrival at the table 31.7 / 36.9 / 42.1 s (Kruskal-Wallis p < 1e-25). A classifier on
  the first-grasp time gives 86 %, on the arrival time 64 %, on the waiting time before the turn-back 55 % (chance 33 %).
  FrameSamp gives the memory the frame index of each remembered frame, so a memory could in principle read these times
  from the images plus time stamps instead of remembering the box. Also the episode length after the decision depends on the
  station (drive back 12.4 / 14.9 / 16.5 s), harmless for the decision. Proposal (no re-record): report the memory result
  next to this time-only ceiling (about 64 % from the table arrival; 86 % only with frame-exact detection of the first grasp),
  and treat memory accuracy clearly above that as real memory. Fix if a re-record is ever needed: also draw the first-phase
  times independently of the station (pad the first pickup to a fixed clock time).

## Fixed

- **F1 Grasp failures (SO-101 tweezer jaws).** Fixed with a 2 cm-wide grip part, a gentle close, grippy
  friction (combine mode max), set-down straight from above and backing off before lifting. Verified 3/3, then 112/120.
- **F2 Conversion crashed on the video-only overview camera.** Fixed: `XLE_VIDEO_CAMERA=0` while converting. Verified: 112 episodes converted.
- **F3 pi0.5 jobs could not import `xle_policies`.** Fixed: repo on `PYTHONPATH` in `slurm/in_container.sh`. Verified by import test.
- **F4 torchcodec needs system FFmpeg.** Fixed: `--dataset.video_backend=pyav`. Verified by training.
- **F5 Hub update of `lerobot/pi05_base` broke loading.** Fixed: pinned revision a538eb27. Verified by training.
- **F6 Memory training stalled on Lustre reads.** Fixed: memory features loaded into RAM. Verified: 5.1 s/step, GPU 100 %.

- **B10 Merged dataset: video timestamps wrong (LeRobot 0.4.2 aggregate_datasets). FIXED 2026-11-26.** Symptom: cache and
  training crash with "query timestamps unexpectedly violate the tolerance" (e.g. 2842.9 s). Cause: when the merge
  starts a new video file, it keeps adding the earlier files' durations to each episode's from/to timestamp (file-001 is
  1432.7 s long but its index went to 2966 s). Videos themselves are fine (frame counts match exactly). Fix:
  `fix_video_timestamps()` in tools/merge_datasets.py recomputes them from episode lengths per file and checks frame
  counts; applied to local/xle_put_it_back_v1 (old table in $HF_LEROBOT_HOME/_archive/v1_meta_episodes_*).
  Verified: 46 frames across all files and boundaries load through LeRobotDataset.
- **B11 FIXED 2026-11-26:** `meta/xle_episodes.json` of local/xle_put_it_back_v1 re-sorted per chunk in the converter's
  text order of demo names; verified length == sim_seconds*30-5 (+-1) for 174/174, stations 58/58/58. Wrong file archived
  in $HF_LEROBOT_HOME/_archive/. tools/merge_datasets.py now sorts the same way.
