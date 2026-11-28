# STATUS (one screen; experiment-planner keeps it current)

_Last update: 2026-11-26 09:15_

## Goal
RSS 2027 Stage 1 extended abstract, due **4 Dec 2026, 23:59 AoE** (10 days). Pilot: pi0.5 without vs.
with memory (FrameSamp + Modul) on PutItBack, same data and settings, only `memory` differs.

## Where we are
- T1 models (D1, 2x2x4 cm block): 0 grasps in 10 eval episodes; fit the demos open-loop; failure is closed loop.
- D1 audit: FAIL visibility (B1), expert commitment (B2), timing shortcut (B3); PASS balance and leak.
- Parity fine. Fixed: B6 (stale first image, `rerender_on_reset`), B8 (pad -3; new runs only via `pad_black`).
  Open, minor: B7 (eval start head pose), B9 (AV1 fades small colours).
- New expert (6 cm box with handle, look-first, random timing, balanced stations, recovery noise):
  X3 clean 3/3, box clearly seen before driving 3/3; noise 1/3 (time-outs, fixed). **X4 (job 29353321, 6 noisy) running.**
- No visible twin of PutItBack exists yet (design gap; Stage 2).

## D2 audit (2026-11-26, data-auditor)
PASS balance 58/58/58 (noise 39+19 per station), visibility (min 10.0x12.1 px), expert commits after seeing, leak at decision (acc 0.30 vs 0.33); FAIL timing (B12: decision frame alone predicts station 65 % vs 33 %); side-file order wrong (B11, labels only). Training can go ahead with the timing caveat.

## D3 audit (2026-11-26, data-auditor) - local/xle_put_it_back_v2, 176 demos, 474,775 frames
- 1 PASS balance: stations 59/59/58 (largest 33.5 %), 176/176 success, noise 41 per station, 0 late, side-file lengths match 176/176.
- 2 PASS visibility: the box covers at least 10.0 x 11.8 px in the 224-px head image when first seen (mean 11.4 x 15.4); per station min 12.5x17.6 / 10.0x13.0 / 10.1x11.8. Close to the 10 px limit, but not under it.
- 3 PASS commit after seeing: the expert starts driving to the station exactly at the moment it sees the box (margin min 0.00 s, never before) and first grasps at least 9.5 s later; 176/176 seen.
- 4 PASS leak: logistic regression (64x64, PCA 40, 5-fold x 5) on the head frame at the decision point: 0.364 (chance 0.333; label-shuffled runs 0.33, max 0.40). At -90 / -30 / +30 frames: 0.36 / 0.34 / 0.30. Wrist cameras 0.27-0.38. Robot state at the decision point: 0.29 (chance 0.33).
- 5 PASS for the decision time (B12 fixed): turn-back time 75.5 / 75.9 / 75.2 s mean by station, Kruskal-Wallis p = 0.71, Spearman -0.03; 1-D classifier 29 % (chance 33 %). Frame index at decision 2036-2516. WARN, other time cues (B13): see below.
- Verdict: training can go ahead. Report the time-only baseline (B13) next to the memory result.
- B13 residual cues (not blocking): events BEFORE the decision still depend on the station because the drive is longer (first grasp 11.9 / 14.4 / 18.2 s; arrival at the table 31.7 / 36.9 / 42.1 s; p < 1e-25). A classifier on the first-grasp time alone reaches 86 %, on the arrival time 64 %, on the time spent waiting at the table before the turn-back 55 % (chance 33 %). The memory sees the frame index of every remembered frame, so in principle it could read these. Episode length is still station-dependent (87.9 / 90.7 / 91.7 s, p = 3e-4, 38 % classifier) but only after the decision, so it cannot help the decision.

## Now (run 3, queued 2026-11-26 09:10)
- D2 not trained (B12). D3 = D2 + timing fix: check 29401783 -> 20 x 9 episodes -> merge 29401824 (stops on a timing leak)
  -> cache -> T3-none 29401826 / T3-mem 29401827 (12k steps, mlgpu_long, ~17 h each) -> evals 50 eps, seed 777.
- Expected: data done ~14:00-16:00 today, trainings done ~27 Nov morning-midday, evals ~27 Nov evening.
- D3 merged 14:35: 176 demos, 474,775 frames, 59/59/58, timing gate 24 % (B12 fixed). Next: data-auditor full audit while queued.

## Deadline checkpoints
- Wed 25 Nov: recording done. Thu 26: audit passed, both trainings submitted.
- Sun 29: T2-none evaluated. Tue 1 Dec: comparison done. Wed 2 Dec 12:00: freeze numbers.
- Fallback: no-memory baseline + task, audits (timing leak B3) as the pilot; comparison as Stage 2.

## Decisions needed from the user
Record D2; parallel vs. sequential T2-mem; repo name for D2; 12k steps; partial-data cutoff; no twin for Stage 1.
