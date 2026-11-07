"""Browser teleop (keyboard + gamepad) for XLeRobot in Isaac Lab, headless on a GPU node.

This is the Isaac Lab twin of xlerobot_sim/teleop_web.py (ManiSkill). It reuses that file's
web page, HTTP server, MJPEG streaming, dead-man switch and keyboard/gamepad mappings
unchanged; only the simulator loop is different:

    browser keys/gamepad -> TeleopCommand (ManiSkill teleop code)
                         -> 17-D action (command_to_action.py, SO-101 IK, real-robot units)
                         -> XLeRobot-Sandbox-v0 in Isaac Lab
                         -> camera frames -> JPEG -> browser

Start it with scripts/start_teleop_isaac.sh (submits the job and sets up the relay), or by
hand inside a GPU job:

    slurm/in_container.sh /isaac-sim/python.sh -m xle_leisaac.teleop.isaac_teleop_web --enable_cameras

--selftest drives a fixed key sequence instead of waiting for a browser, checks that the
robot moved the right way, saves the frames and exits.
"""

import argparse
import os
import sys
import secrets
import socket
import threading
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="XLeRobot-Sandbox-v0")
parser.add_argument("--port", type=int, default=None, help="default: 20000 + uid % 20000, next free one if taken")
parser.add_argument("--bind", default="0.0.0.0")
parser.add_argument("--max_fps", type=float, default=30.0)
parser.add_argument("--jpeg_quality", type=int, default=80)
parser.add_argument("--deadzone", type=float, default=0.15)
parser.add_argument("--token", default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--selftest", action="store_true", help="scripted keys, no browser; prints PASS/FAIL")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import xle_leisaac  # noqa: F401, E402  (registers XLeRobot-Sandbox-v0)
from xle_leisaac.robots.xlerobot import REST_ARM_POSE  # noqa: E402
from xle_leisaac.teleop.command_to_action import JOINTS, IsaacTeleopAdapter  # noqa: E402  (also loads xlerobot_sim)
from xlerobot_sim import teleop_web as web  # noqa: E402
from xlerobot_sim.teleop_controller import GAMEPAD_HELP, KEY_HELP, gamepad_command, keyboard_command  # noqa: E402

# Extra controls the Isaac version adds (the real base can strafe; real speed levels).
EXTRA_KEYS = [("Isaac only", [("Q / E", "base strafe left / right"), ("1 / 2 / 3", "speed slow / medium / fast")])]
EXTRA_PAD = [("Isaac only", [("View + D-pad left/right", "base strafe left / right")])]

# Third-person camera that follows the robot from behind (as in the ManiSkill page).
CHASE_CAMERA = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_link/chase_camera",
    offset=TiledCameraCfg.OffsetCfg(pos=(-1.6, 0.0, 1.7), rot=(0.9659, 0.0, 0.2588, 0.0), convention="world"),
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, clipping_range=(0.05, 30.0)),
    width=960,
    height=540,
)
STREAMS = {"main": "chase", "head": "head", "wrist1": "right_wrist", "wrist2": "left_wrist"}  # page name -> sensor


def build_page(token):
    return (web.PAGE_TEMPLATE.replace("__TOKEN__", token)
            .replace("__KEYS__", web._help_rows(KEY_HELP + EXTRA_KEYS))
            .replace("__PAD__", web._help_rows(GAMEPAD_HELP + EXTRA_PAD)))


def start_server(state):
    token = args.token or secrets.token_urlsafe(12)
    handler = web.make_handler(state, token, build_page(token))
    first = args.port or 20000 + os.getuid() % 20000
    for port in range(first, first + 20):
        try:
            server = web.ThreadingHTTPServer((args.bind, port), handler)
            break
        except OSError:
            continue
    else:
        raise SystemExit(f"no free port in {first}..{first + 19}")
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host = socket.gethostname().split(".")[0]
    # The same lines as the ManiSkill server: scripts/start_teleop.sh looks for them.
    print("=" * 72)
    print(f" Teleop server on {host}:{port}")
    print(f" Open: http://localhost:{port}/?token={token}")
    print("=" * 72, flush=True)
    return server


def selftest_input(step):
    """(pressed keys, newly pressed keys) for the scripted self-test."""
    plan = [(0, 60, {"w"}), (60, 120, {"q"}), (120, 180, {"a"}), (180, 240, {"9"}), (240, 300, {"8"}), (300, 330, {"r"})]
    pressed = next((keys for start, end, keys in plan if start <= step < end), set())
    edges = {"v"} if step == 330 else set()
    return pressed, edges


def main():
    state = web.SharedState()
    server = None if args.selftest else start_server(state)

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.seed = args.seed
    env_cfg.terminations.time_out = None  # teleop runs until you stop it
    env_cfg.scene.chase = CHASE_CAMERA
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    obs, _ = env.reset()
    robot = env.scene["robot"]
    rest = [REST_ARM_POSE[j] for j in JOINTS]

    def current_state():
        return obs["policy"]["state"][0].cpu().numpy()

    adapter = IsaacTeleopAdapter(current_state(), rest)
    print(f"[teleop] env ready: {args.task}, action dim {env.action_manager.total_action_dim}", flush=True)

    enc = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
    dt_min = 1.0 / args.max_fps
    t_prev, fps, step = time.time(), 0.0, 0
    prev_pad_buttons = []
    start_pose = robot.data.joint_pos[0, :3].clone()
    try:
        while simulation_app.is_running():
            t0 = time.time()
            if args.selftest:
                pressed, edges = selftest_input(step)
                pad, do_reset = None, False
                if step >= 360:
                    break
            else:
                with state.lock:
                    alive = t0 - state.last_input < 1.0  # dead-man switch: browser went quiet -> stop
                    pressed = set(state.pressed) if alive else set()
                    pad = state.gamepad if alive else None
                    edges, state.edges = state.edges, set()
                    do_reset, state.reset_requested = state.reset_requested, False
            if do_reset:
                obs, _ = env.reset()
                adapter = IsaacTeleopAdapter(current_state(), rest)

            cmd = keyboard_command(pressed, edges)
            pad_info = None
            if pad and pad.get("mapping") == "standard":
                cmd = cmd + gamepad_command(pad["axes"], pad["buttons"], prev_pad_buttons, args.deadzone)
                prev_pad_buttons = pad["buttons"]
                pad_info = pad.get("id", "gamepad")
            elif pad:
                pad_info = f"unsupported layout ({pad.get('mapping') or 'non-standard'}): {pad.get('id')}"

            action = adapter.step(cmd, pressed=pressed, edges=edges, gamepad=pad)
            with torch.inference_mode():
                obs, *_ = env.step(torch.as_tensor(action, device=env.device).unsqueeze(0))
            step += 1

            frames = {name: web.to_uint8_image(env.scene[sensor].data.output["rgb"]) for name, sensor in STREAMS.items()}
            jpgs = {k: cv2.imencode(".jpg", cv2.cvtColor(v, cv2.COLOR_RGB2BGR), enc)[1].tobytes() for k, v in frames.items()}
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now
            with state.frame_cond:
                state.frames = jpgs
                state.frame_id += 1
                state.status = {
                    "fps": round(fps, 1),
                    "step": step,
                    "warmup": False,
                    "pressed": sorted(pressed),
                    "gamepad": pad_info,
                    "grippers_closed": adapter.gripper_closed,
                    "joints": np.round(current_state(), 3).tolist(),
                }
                state.frame_cond.notify_all()

            if args.selftest and step in (1, 59, 119, 179, 239, 299, 359):
                out = os.path.join(os.environ["XLE_LOGS"], "teleop_selftest")
                os.makedirs(out, exist_ok=True)
                for name, image in frames.items():
                    cv2.imwrite(os.path.join(out, f"{step:03d}_{name}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            if not args.selftest:
                sleep = dt_min - (time.time() - t0)
                if sleep > 0:
                    time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    finally:
        state.running = False
        if server:
            server.shutdown()

    if args.selftest:
        s = current_state()
        moved = (robot.data.joint_pos[0, :3] - start_pose).cpu().numpy()
        checks = {
            "W drove forward (+x)": moved[0] > 0.1,
            "Q strafed left (+y)": moved[1] > 0.1,
            "A turned left (+yaw)": moved[2] > 0.5,
            "9/8 moved right arm (lift or elbow changed)": abs(s[7] - rest[1]) + abs(s[8] - rest[2]) > 0.1,
            "left arm stayed put (no keys for it)": abs(s[1] - rest[1]) + abs(s[2] - rest[2]) < 0.02,
            "R panned head": s[12] > 0.3,
            "V closed right gripper": s[11] < 0.2,
        }
        print(f"[teleop] base moved x={moved[0]:.3f} m y={moved[1]:.3f} m yaw={np.degrees(moved[2]):.1f} deg; fps {fps:.1f}")
        for name, ok in checks.items():
            print(f"[teleop]   {'ok  ' if ok else 'FAIL'} {name}")
        print(f"[teleop] SELFTEST {'PASSED' if all(checks.values()) else 'FAILED'}")
    os._exit(0)  # Kit can hang on shutdown; nothing here needs a clean exit


if __name__ == "__main__":
    # Fail fast: if anything raises, print it and exit now. Otherwise Kit can hang on
    # shutdown and the job sits idle until its time limit.
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
