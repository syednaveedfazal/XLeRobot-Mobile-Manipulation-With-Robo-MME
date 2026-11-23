"""calibrate_camera.py - measure the real XLeRobot cameras so the sim cameras can match them.

Runs on the robot's computer (not Marvin): pip install opencv-python numpy  (+ pyrealsense2
for the head camera). Each mode prints its result and saves it as JSON; send the JSON files.

1. Head camera (RealSense): factory calibrated, just read it out (also tells D435 vs D455):
     python tools/calibrate_camera.py realsense --width 640 --height 480

2. Wrist cameras (UVC, no factory data): photograph a checkerboard, then solve.
   Print a checkerboard with 9 x 6 INNER corners (10 x 7 squares), measure one square with a
   ruler, glue it flat on cardboard. Use the resolution the robot records with.
     python tools/calibrate_camera.py capture --camera 0 --out calib_right_wrist --width 640 --height 480
       (SPACE saves a photo, Q quits. Take ~20: board near/far, tilted, in every corner of the image.)
     python tools/calibrate_camera.py intrinsics --images calib_right_wrist --square 0.025

3. Where a camera sits (mainly the head camera): put the robot's head at its calibration pose
   (pan 0, tilt 0), lay a checkerboard flat on the floor in view (bigger is better here: print
   it on A3 with 4 cm squares, still 9 x 6 inner corners), its rows along the robot's
   forward direction, and measure where its first inner corner is, in metres from the centre of
   the robot's base on the floor: x forward, y left. Take one photo and solve:
     python tools/calibrate_camera.py capture --camera 4 --out head_pose --width 640 --height 480
     python tools/calibrate_camera.py pose --image head_pose/000.png --intrinsics head_intrinsics.json \
         --corner 0.80 0.20 --square 0.04
   It saves pose_check.png with the first corner circled (red) and the board's x axis (green):
   check that red is the corner you measured and green points forward, or turn the board.
"""

import argparse
import glob
import json
import math
import os

import numpy as np


def board_points(cols, rows, square):
    grid = np.zeros((rows * cols, 3), np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return grid


def find_corners(cv2, image, cols, rows):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCorners(gray, (cols, rows), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    # Refine to sub-pixel, searching less than half a square around each corner (a fixed
    # window would jump to the next corner when the board is small or seen at a slant).
    grid = corners.reshape(rows, cols, 2)
    spacing = min(np.linalg.norm(np.diff(grid, axis=0), axis=-1).min(), np.linalg.norm(np.diff(grid, axis=1), axis=-1).min())
    half = int(max(2, min(11, 0.4 * spacing)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    return cv2.cornerSubPix(gray, corners, (half, half), (-1, -1), criteria)


def save(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(json.dumps(data, indent=2))
    print(f"saved {path}")


def cmd_realsense(a):
    import pyrealsense2 as rs

    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.color, a.width, a.height, rs.format.bgr8, 30)
    profile = pipe.start(cfg)
    device = profile.get_device()
    k = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    pipe.stop()
    save(a.out or "head_intrinsics.json", dict(
        camera=device.get_info(rs.camera_info.name), serial=device.get_info(rs.camera_info.serial_number),
        width=k.width, height=k.height, fx=k.fx, fy=k.fy, cx=k.ppx, cy=k.ppy, distortion=list(k.coeffs),
        hfov_deg=math.degrees(2 * math.atan(k.width / (2 * k.fx))), vfov_deg=math.degrees(2 * math.atan(k.height / (2 * k.fy))),
    ))  # fmt: skip


def cmd_capture(a):
    import cv2

    os.makedirs(a.out, exist_ok=True)
    cap = cv2.VideoCapture(a.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    n = len(glob.glob(os.path.join(a.out, "*.png")))
    print("SPACE: save   Q: quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            raise SystemExit(f"cannot read camera {a.camera}")
        view = frame.copy()
        corners = find_corners(cv2, frame, a.cols, a.rows)
        if corners is not None:
            cv2.drawChessboardCorners(view, (a.cols, a.rows), corners, True)
        cv2.putText(view, f"{n} saved  {frame.shape[1]}x{frame.shape[0]}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("calibrate", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            cv2.imwrite(os.path.join(a.out, f"{n:03d}.png"), frame)
            n += 1
        elif key == ord("q"):
            break
    cap.release()


def cmd_intrinsics(a):
    import cv2

    objects, images, size = [], [], None
    for path in sorted(glob.glob(os.path.join(a.images, "*.png"))):
        image = cv2.imread(path)
        size = image.shape[1], image.shape[0]
        corners = find_corners(cv2, image, a.cols, a.rows)
        print(f"{path}: {'ok' if corners is not None else 'board not found, skipped'}")
        if corners is not None:
            objects.append(board_points(a.cols, a.rows, a.square))
            images.append(corners)
    if len(images) < 8:
        raise SystemExit(f"only {len(images)} usable photos; take at least 8 (20 is better)")
    rms, k, dist, _, _ = cv2.calibrateCamera(objects, images, size, None, None, flags=cv2.CALIB_RATIONAL_MODEL)
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    print(f"reprojection error {rms:.2f} px (good: below 0.5-1.0)")
    save(a.out or os.path.basename(os.path.normpath(a.images)) + "_intrinsics.json", dict(
        width=size[0], height=size[1], fx=fx, fy=fy, cx=cx, cy=cy, distortion=dist.ravel().tolist(), rms_px=rms,
        photos=len(images), hfov_deg=math.degrees(2 * math.atan(size[0] / (2 * fx))),
        vfov_deg=math.degrees(2 * math.atan(size[1] / (2 * fy))),
    ))  # fmt: skip


def cmd_pose(a):
    import cv2

    with open(a.intrinsics) as f:
        intr = json.load(f)
    k = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]])
    dist = np.array(intr["distortion"])
    image = cv2.imread(a.image)
    corners = find_corners(cv2, image, a.cols, a.rows)
    if corners is None:
        raise SystemExit("checkerboard not found in the photo")
    # Board on the floor: its columns (OpenCV x) along robot forward, rows toward robot right.
    grid = board_points(a.cols, a.rows, a.square)
    world = np.stack([a.corner[0] + grid[:, 0], a.corner[1] - grid[:, 1], np.zeros(len(grid))], axis=1)
    # A checkerboard looks the same turned 180 degrees, so OpenCV may number the corners from
    # either end. Solve both ways and keep the one that puts the camera on the robot (nearest
    # the base centre), i.e. the board is where you measured it.
    best = None
    for ordered in (corners, corners[::-1].copy()):
        ok, rvec, tvec = cv2.solvePnP(world, ordered, k, dist)
        rot = cv2.Rodrigues(rvec)[0]
        position = (-rot.T @ tvec).ravel()  # camera centre in the robot frame
        if best is None or np.hypot(*position[:2]) < np.hypot(*best[3][:2]):
            best = (ordered, rvec, tvec, position, rot)
    corners, rvec, tvec, position, rot = best
    forward, down = rot.T @ np.array([0, 0, 1.0]), rot.T @ np.array([0, 1.0, 0])  # optical axis, image-down
    error = np.linalg.norm(cv2.projectPoints(world, rvec, tvec, k, dist)[0].reshape(-1, 2) - corners.reshape(-1, 2), axis=1).mean()
    check = image.copy()
    first = tuple(int(v) for v in corners[0, 0])
    along = tuple(int(v) for v in corners[1, 0])
    cv2.circle(check, first, 10, (0, 0, 255), 3)
    cv2.arrowedLine(check, first, along, (0, 255, 0), 3, tipLength=0.4)
    cv2.imwrite("pose_check.png", check)
    save(a.out or "camera_pose.json", dict(
        position_m=position.tolist(), optical_axis=forward.tolist(), image_down=down.tolist(),
        pitch_down_deg=math.degrees(math.asin(-forward[2])), yaw_deg=math.degrees(math.atan2(forward[1], forward[0])),
        reprojection_px=float(error), image=a.image, board_first_corner=a.corner, square=a.square,
    ))  # fmt: skip
    print("check pose_check.png: red circle = the corner you measured, green arrow = robot forward")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    for name in ("realsense", "capture", "intrinsics", "pose"):
        s = sub.add_parser(name)
        s.add_argument("--cols", type=int, default=9, help="inner corners per row")
        s.add_argument("--rows", type=int, default=6, help="inner corners per column")
        s.add_argument("--out", default=None)
        if name in ("realsense", "capture"):
            s.add_argument("--width", type=int, default=640)
            s.add_argument("--height", type=int, default=480)
        if name == "capture":
            s.add_argument("--camera", type=int, default=0, help="OpenCV camera index (/dev/videoN)")
        if name in ("intrinsics", "pose"):
            s.add_argument("--square", type=float, required=True, help="checkerboard square size in metres")
        if name == "intrinsics":
            s.add_argument("--images", required=True)
        if name == "pose":
            s.add_argument("--image", required=True)
            s.add_argument("--intrinsics", required=True)
            s.add_argument("--corner", type=float, nargs=2, required=True, metavar=("X", "Y"),
                           help="first inner corner on the floor, metres from the base centre: x forward, y left")
    a = p.parse_args()
    {"realsense": cmd_realsense, "capture": cmd_capture, "intrinsics": cmd_intrinsics, "pose": cmd_pose}[a.mode](a)


if __name__ == "__main__":
    main()
