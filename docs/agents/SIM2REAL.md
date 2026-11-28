# SIM2REAL (sim2real-engineer)

## Calibration (single source of truth)
- Arms: `calibration/my_follower.json` (follower) and `calibration/my_leader.json` (leader), used by
  `xle_leisaac/robots/calibration.py`; sim zero = calibration pose (2047 ticks). Date: see git history.
- Head camera: model assumed RealSense D435 (69 deg HFOV), 640x360 in sim. Intrinsics/extrinsics: NOT MEASURED.
- Wrist cameras: 120 deg HFOV, 640x360, pose from the official SO-101 wrist mount. Intrinsics: NOT MEASURED.
- Tool for measuring: `tools/calibrate_camera.py` (realsense / capture / intrinsics / pose).

## Scene (sim values the real setup should match)
- 3 identical stations, 14x14 cm top, 70 cm high, at x = 0.70 / 1.20 / 1.70 m ahead, 0.35 m to the right of the start.
- Work table 60x40 cm top, 70 cm high, centre 0.90 m behind the start, 0.47 m to the left; drop spot 0.35 m left.
- Object (in progress): 6 cm box with a red 2x2x4 cm handle, ~80 g. Grip tape on the jaws (sim uses friction 1.5, combine mode max).

## Measured gaps sim vs. real
- None measured yet.

## Notes
- Applying real camera calibration to the sim cameras changes the images: every dataset recorded before
  that no longer matches.
