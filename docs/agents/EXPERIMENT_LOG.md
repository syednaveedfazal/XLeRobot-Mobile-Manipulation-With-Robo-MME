# EXPERIMENT_LOG (training-ops and eval-analyst add one entry per run)

Format: ID | date | Slurm job | what changed vs. which run | config / paths | result | conclusion

## Data

- **D1** | 2026-11-21 | jobs 29246357-69 (record), 29271795 (convert) | first PutItBack dataset |
  object 2x2x4 cm block, 120 episodes recorded, random stations, no noise; dataset
  `$HF_LEROBOT_HOME/local/xle_put_it_back_v0` | 112/120 successful demos kept, 192,520 frames |
  usable for training, but the block is only 2-3 px at 224 px (see BUGS.md B1).

## Training

- **T1-none** | 2026-11-24 | job 29304686 | first run (baseline) | pi0.5 `memory=none`, dataset D1,
  12k steps, batch 16, lr 2.5e-5 warmup/cosine, pinned `lerobot/pi05_base` rev a538eb27,
  `slurm/train_pi05_mem.sbatch`; checkpoints `$XLE_WORK/train_runs/pi05_none_xle_put_it_back_v0_29304686/` |
  final loss 0.006, 16 h 32 min (4.87 s/step) | trained fine.
- **T1-mem** | 2026-11-24 | job 29304688 | vs. T1-none: memory `framesamp_modul` only | same settings;
  checkpoints `$XLE_WORK/train_runs/pi05_framesamp_modul_xle_put_it_back_v0_29304688/` |
  final loss 0.005, 17 h 19 min (5.10 s/step) | trained fine.

## Evaluation

- **E1** | 2026-11-24 | jobs 29352128 (smoke), 29352129, 29352130 (cancelled) | T1-none and T1-mem,
  seed 777, `tools/eval_put_it_back.py` | 0/4 (none) and 0/5 + 0/1 (memory): never picked up the
  block; drove to the wrong station or overshot | memory comparison not possible yet.
- **E2** | 2026-11-24 | job 29352704 | vs. E1: plan every 10 steps instead of 50 (T1-none) | 0/3 |
  chunk length is not the cause.
- **E3** | 2026-11-24 | job 29352703 | open-loop on training episode 5 (T1-none) | mean |error|:
  right arm 0.87, gripper 0.89, head 0.44 (scale -100..100) | the model fits the demos; failure is closed loop.

## Expert (data generation) tests

- **X1** | 2026-11-24 | job 29353206 (cancelled after 5 eps) | vs. D1 expert: 6 cm box with handle, look-first,
  lifts 6 cm | clean 3/3 success (lift 5.7 cm); "clearly seen" only 1/3 (bug B4); with noise 0/2 (bug B5) |
  grasp on the handle works; visibility check and noise needed fixes.
- **X2** | 2026-11-24 | job 29353270 | vs. X1: B5 fix, random timing (B3), first B4 attempt (camera flag; did not work) | clean 3/3, seen 1/3 |
- **X3** | 2026-11-24 | job 29353289 | vs. X2: B4 fixed properly (head camera pose from the robot link) | (running) |
  X3 result: clean 3/3 success, box clearly seen before driving 3/3 (13x18 / 10x13 / 10x15 px); with noise 1/3, the
  2 failures were time-outs (100 s), not grasp failures: base noise near the goal slowed every stop.
- **X4** | 2026-11-24 | job 29353321 | vs. X3: base noise off within 8 cm / 0.15 rad of a base goal; episode limit 120 s | noise 6/6 success, box clearly seen before driving 6/6 | recovery noise works.
- **X5** | 2026-11-24 | job 29353344 | vs. X4: room added (4 walls, shelf, sofa, cabinet, door; overview camera moved inside) | clean 3/3, noise 3/3, box clearly seen before driving 6/6 | approach ready for recording (pending the user's OK).

## Run 2 (queued 2026-11-24 22:27, user approved; deadline 25 Nov evening)

- **D2** | jobs 29353506 (check), record/convert 15 chunks, 29353537 (merge) | vs. D1: 6 cm box with handle, look-first
  expert, random timing, balanced stations, 2/3 chunks with recovery noise, room, rerender on reset | 15 x 12 = 180
  episodes -> `local/xle_put_it_back_v1` (+ meta/xle_episodes.json) | (running)
- **T2-none** | job 29353539 | vs. T1-none: dataset D2, pad_black, 5k steps (deadline), save every 2.5k | (queued)
- **T2-mem** | job 29353540 | vs. T2-none: memory framesamp_modul only (cache job 29353538, pad_black) | (queued)
- **E4** | queued 2026-11-24 | eval jobs after T2-none / T2-mem (afterok), 20 episodes each, seed 777, 4 videos |
  results in `$XLE_LOGS/eval_put_it_back/v1_none_5k/` and `v1_mem_5k/` (results.json + videos) | (queued)
- **Run 2 status 2026-11-25 18:20:** recording done (172/180 episodes succeeded). Convert of chunks 5, 10, 13, 14 failed:
  Lustre itself was full ("No space left on device"; cluster file system 95-96 %, metadata 98 %) while 15
  conversions wrote temporary PNG frames at once. Merge failed on those broken chunks (LeRobot then asked the Hub).
  Fixes: merge skips unreadable chunks and never asks the Hub; the 4 chunks reconverted 2 at a time
  (jobs 29387251-54); merge 29387255; cache 29387256; T2-none 29387257; T2-mem 29387258; evals queued after training.
  Freed ~140 GB: deleted T1 step-6000 checkpoints and T1 optimizer states (final T1 weights kept).
- **Run 2 update 2026-11-25 19:20:** user asked for 12k steps ("should be perfect") and A100s. Trainings moved to
  A100 (sgpu_short, 8 h; training needs no Isaac Sim) with a resume job each in case 8 h is not enough
  (`RESUME_RUN` in slurm/train_pi05_mem.sbatch): T2-none 29387774 (+resume 29387775), T2-mem 29387777 (+resume
  29387778); cache 29387256 on sgpu_devel; conversions and merge on mlgpu_devel (A40 queue: 8 h partition
  estimated tomorrow 09:40, 1-day partition 28 Nov). Evals stay on A40 (Isaac rendering): 50 episodes each,
  seed 777: 29387776 (none), 29387779 (memory). Checkpoints every 4k steps (disk is ~96 % full cluster-wide).
- **Run 2 update 2026-11-26 08:30:** merge OK (174 demos, 339,242 frames, stations 58/58/58). Cache and T2-none crashed:
  LeRobot 0.4.2 aggregate_datasets kept adding earlier files' durations to the episodes' video timestamps (bug B10).
  Fixed in tools/merge_datasets.py (fix_video_timestamps) and applied to the dataset (old table archived); 46 frames
  across all files and boundaries load. Resubmitted on A40 mlgpu_long (A100 queue now until 27 Nov):
  cache 29400764, T2-none 29400765, T2-mem 29400766 (12k steps, 20 h jobs), evals after them (50 episodes, seed 777).
- **Run 2 stopped 2026-11-26 09:00 (user):** D2 has the timing shortcut B12, so its trainings were cancelled before they
  started (cache 29400764, T2-none 29400765, T2-mem 29400766, evals 29400767/68). D2 is kept until D3 has merged.

## Run 3 (queued 2026-11-26 09:10, user asked: re-record with the timing fix, no more training problems)

- **D3** | jobs 29401783 (check), 29401784-823 (record + convert, 20 chunks), 29401824 (merge) | vs. D2: one change, the
  B12 timing fix (table wait until a random clock time 60-75 s; episode limit 150 s) | 20 x 9 = 180 episodes, 2/3 with
  recovery noise, stations cycled -> `local/xle_put_it_back_v2` | (running)
- **T3-none** | job 29401826 (resume 29401828 only if it stops early) | vs. T2-none: dataset D3 | 12k steps, batch 16,
  save every 6k, mlgpu_long 22 h, pad_black | (queued)
- **T3-mem** | job 29401827 (cache 29401825; resume 29401831) | vs. T3-none: memory framesamp_modul only | 128 GB RAM | (queued)
- **E5** | jobs 29401829/30 (none), 29401832/33 (memory; the second of each only after a resume) | 50 episodes, seed 777,
  4 videos; results saved after every episode | `$XLE_LOGS/eval_put_it_back/v2_none_12k/` and `v2_mem_12k/` | (queued)
- Disk: deleted D1 (`local/xle_put_it_back_v0`, 40 GB), the T1 model folders (38 GB; results are in E1-E3) and the
  Hugging Face download chunk cache (`cache/huggingface/xet`, 23 GB; pi05_base itself kept). Each raw chunk is now
  deleted right after its conversion is verified, at most 3 conversions run at once, and D2 is deleted after a good merge.
- **Run 3 status 2026-11-26 12:50:** check 29401783 passed (6/6 with noise, none late; turn-back 8-9 s after the drawn
  time for every station). Recording done: 176/180 successful (4 time-outs at 150 s: 3 finished the plan but the box
  did not count as placed, 1 stuck driving to the station; the converter drops them), 0 late. Timing gate on all 176
  side-file entries: turn-back 75.5 / 75.9 / 75.2 s for station 0/1/2, station from time alone 31 % (chance 33 %): B12
  fixed. Episode length mean 90 s, max 105 s. Conversions 9/20 done, 3 at a time; training estimate mlgpu_long 27 Nov ~05:30.

- **Run 3 update 2026-11-26 13:00 (user: "use whatever gpu available"):** trainings and resumes may now start on A40 or
  A100 (mlgpu_long/medium, sgpu_long/medium; training needs no Isaac rendering), cache and merge also on A100, evals on
  any A40 partition (Isaac rendering). Conversions 6 at a time instead of 3 (cluster disk 93 %, 350 TB free).
  Hardware is not part of the comparison: whichever GPU each run gets is noted with its result.
- **Run 3 update 2026-11-26 14:35:** all 20 conversions OK; merge 29401824 OK: D3 = `local/xle_put_it_back_v2`, 176 demos,
  474,775 frames, stations 59/59/58; timing gate 24 % (chance 33 %), 0 late; side file order checked (176/176 episode
  lengths match within 1 frame). Raw chunks, per-chunk datasets and D2 deleted; our disk use 109 GB. Cache 29401825
  running (~1 h); T3-none estimated start 27 Nov 01:39.
- **Run 3 update 2026-11-26 15:45:** cache 29401825 done (474,775 frames, 58 min). D3 audit (data-auditor): checks 1-5 PASS;
  new B13 (low): times before the decision (first grasp, table arrival) still depend on the station, readable only by a
  memory model through frame times; report T3-mem next to that time-only ceiling, and plan a "random pauses" eval.
  22 h jobs were estimated to start 27 Nov 06:00-12:35, 8 h jobs tonight, so the trainings were resubmitted as chains of
  8 h jobs on A40 or A100 (mlgpu_short, sgpu_short): first job + up to 3 resumes, checkpoint every 2500 steps, each
  resume deletes all but the last checkpoint. Settings otherwise unchanged (12k steps, batch 16, pad_black).
  T3-none 29408513 -> 29408514 -> 29408516 -> 29408518, evals 29408515/17/19 (the one after the finishing job runs);
  T3-mem 29408520 -> 29408521 -> 29408523 -> 29408525, evals 29408522/24/26. The chains continue only if the resume
  test 29408507 passes (40 steps, killed after the step-20 checkpoint, then resumed to 40).
- **Resume test 29408507 PASSED 2026-11-26 16:10:** killed after the step-20 checkpoint, resumed at step 21 (lr schedule continued) to 40; a second resume saw it finished and exited. Test checkpoints deleted. The chains can continue.
