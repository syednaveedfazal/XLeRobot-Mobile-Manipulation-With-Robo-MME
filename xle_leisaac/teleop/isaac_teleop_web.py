"""Browser teleop (keyboard + gamepad) for XLeRobot in Isaac Lab, headless on a GPU node.

This is the Isaac Lab twin of xlerobot_sim/teleop_web.py (ManiSkill). It reuses that file's
web page, HTTP server, MJPEG streaming, dead-man switch and keyboard/gamepad mappings
unchanged; only the simulator loop is different:

    browser keys/gamepad -> TeleopCommand (ManiSkill teleop code)
                         -> 17-D action (command_to_action.py, SO-101 IK, real-robot units)
                         -> XLeRobot-Sandbox-v0 in Isaac Lab
                         -> camera frames -> JPEG -> browser

It also speaks the real robot's language, so real-robot teleop code can drive the sim:
    POST /action       {"action": {"left_arm_shoulder_pan.pos": ..., "x.vel": ..., ...}}
                       the real XLeRobot's action dict (LeRobot units); converted with the
                       arm calibration. Takes priority over keyboard/gamepad while it keeps
                       arriving (tools/joycon_sim_client.py sends it from XLeRobot's Joy-Con code).
    GET  /observation  the sim's state as the real robot's observation dict (LeRobot units).

Start it with scripts/start_teleop_isaac.sh (submits the job and sets up the relay), or by
hand inside a GPU job:

    slurm/in_container.sh /isaac-sim/python.sh -m xle_leisaac.teleop.isaac_teleop_web --enable_cameras

An SO-101 leader arm can drive one or both sim arms: run scripts/leader_client.py on the
computer the leader is plugged into. Its servo readings are posted to /leader and turned
into joint angles with LeRobot calibration files (leader_arm.py). While leader data
arrives, that arm follows the leader; when it stops for 0.5 s, keys and gamepad take over
again from the current pose.

--policy <checkpoint> lets a trained pi0.5 policy drive instead of keys/gamepad (needs XLE_WITH_PI=1;
scripts/watch_policy.sh sets it all up). It gets what it was trained on (head + wrist images, state,
task text), episodes end as in training, and the next one starts by itself with a fresh memory.

--expert (PutItBack only) lets the scripted expert drive, exactly as when it records demos
(--noise adds the recovery-demo action noise); scripts/watch_expert.sh sets it up.

--selftest drives a fixed key sequence instead of waiting for a browser, then a scripted
Xbox gamepad sequence and a leader-arm pose, checks that the robot moved the right way,
saves the frames and exits.
"""

import argparse
import json
import os
import sys
import secrets
import socket
import threading
import time
from urllib.parse import urlparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="XLeRobot-Sandbox-v0")
parser.add_argument("--port", type=int, default=None, help="default: 20000 + uid % 20000, next free one if taken")
parser.add_argument("--bind", default="0.0.0.0")
parser.add_argument("--max_fps", type=float, default=30.0)
parser.add_argument("--jpeg_quality", type=int, default=92, help="stream quality; higher is sharper, more bandwidth")
parser.add_argument("--chase_size", type=int, nargs=2, default=(1280, 720), metavar=("W", "H"),
                    help="chase-camera resolution (the robot's own cameras stay as the policy sees them)")
parser.add_argument("--deadzone", type=float, default=0.15)
parser.add_argument("--token", default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--selftest", action="store_true", help="scripted keys, no browser; prints PASS/FAIL")
parser.add_argument("--policy", default=None, help="checkpoint .../pretrained_model: a trained policy drives the robot")
parser.add_argument("--expert", action="store_true", help="the PutItBack scripted expert drives the robot")
parser.add_argument("--noise", action="store_true", help="with --expert: the recovery-demo action noise")
parser.add_argument("--leader_calibration", default="calibration/my_leader.json",
                    help="SO-101 leader arm calibration (LeRobot format), relative to the repo. The sim arms "
                         "use the follower calibration in xle_leisaac/robots/calibration.py (ARM_CALIBRATION_FILES)")
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
from isaaclab.managers import RecorderManagerBaseCfg  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import xle_leisaac  # noqa: F401, E402  (registers XLeRobot-Sandbox-v0)
from xle_leisaac.robots.xlerobot import (  # noqa: E402
    ARM_CALIBRATION,
    REST_ARM_POSE,
    XLEROBOT_ACTION_KEYS,
    lerobot_to_sim_action,
    sim_to_lerobot_action,
)
from xle_leisaac.teleop.command_to_action import JOINTS, REPO, IsaacTeleopAdapter  # noqa: E402  (also loads xlerobot_sim)
from xle_leisaac.teleop.leader_arm import HOMED_TICKS, TICKS_PER_TURN, LeaderToSim, load_calibration  # noqa: E402
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


REAL_ACTION_TIMEOUT = 0.5  # s without a new /action -> back to keyboard/gamepad


def make_handler(state, token, page):
    """The ManiSkill page's HTTP handler, plus /action and /observation for real-robot code."""
    base = web.make_handler(state, token, page)

    class Handler(base):
        def do_GET(self):
            if urlparse(self.path).path == "/observation":
                if self._authorized():
                    with state.lock:
                        body = json.dumps(state.observation).encode()
                    self._send(200, body, "application/json")
                return
            super().do_GET()

        def do_POST(self):
            if urlparse(self.path).path == "/action":
                if self._authorized():
                    data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                    with state.lock:
                        state.real_action = data.get("action", {})
                        state.real_action_time = time.time()
                        state.real_action_source = data.get("source", "client")
                    self._send(200, b"{}", "application/json")
                return
            super().do_POST()

    return Handler


def start_server(state):
    token = args.token or secrets.token_urlsafe(12)
    handler = make_handler(state, token, build_page(token))
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


# Self-test pose for the leader arm (degrees; gripper in % open), sent as servo ticks.
SELFTEST_LEADER_DEG = {"shoulder_pan": 30.0, "shoulder_lift": -40.0, "elbow_flex": 30.0,
                       "wrist_flex": 20.0, "wrist_roll": 45.0, "gripper": 50.0}
LEADER_TIMEOUT = 0.5  # s without leader data -> keys/gamepad control that arm again


# Scripted Xbox input (W3C standard layout, as the browser sends it): (start, end, axes, buttons held).
# axes = [LX, LY, RX, RY], stick up = -1. Button indices: A0 B1 X2 Y3 LB4 RB5 LT6 RT7 L3 10 R3 11 Up12.
PAD_PHASES = (
    (360, 400, [1.0, 0.0, 0.0, -1.0], {6}),   # left stick right: left EE forward; right stick up: right EE up; LT: close left gripper
    (400, 440, [0.0, 0.0, 1.0, 0.0], {11, 1, 12}),  # R3 + right stick: right shoulder pan; B: head pan right; D-pad up: forward
    (440, 480, [0.0, -1.0, 0.0, 0.0], {4}),   # LB + left stick up: left wrist pitch
)
LEADER_START, SELFTEST_END = 480, 570


def selftest_gamepad(step):
    for start, end, axes, held in PAD_PHASES:
        if start <= step < end:
            return {"id": "selftest Xbox", "mapping": "standard", "axes": axes,
                    "buttons": [1.0 if i in held else 0.0 for i in range(17)]}
    return None


def selftest_leader_ticks(cal):
    ticks = []
    for joint in JOINTS:
        value = SELFTEST_LEADER_DEG[joint]
        if joint == "gripper":
            c = cal[joint]
            ticks.append(c["range_min"] + value / 100.0 * (c["range_max"] - c["range_min"]))
        else:  # degrees from the calibration pose
            ticks.append(HOMED_TICKS + value * TICKS_PER_TURN / 360.0)
    return ticks


def main():
    state = web.SharedState()
    state.real_action, state.real_action_time, state.real_action_source = {}, 0.0, None
    state.observation = {}
    server = None if args.selftest else start_server(state)

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.seed = args.seed
    if not (args.policy or args.expert):
        env_cfg.terminations.time_out = None  # teleop runs until you stop it; policy/expert episodes end as usual
    # No recording here. The env's recorder keeps every step (camera images, the whole scene
    # state) in memory until the episode ends, and a teleop episode never ends: memory grew
    # ~2 GB/min until Slurm killed the job (OUT_OF_MEMORY) and the robot froze.
    env_cfg.recorders = RecorderManagerBaseCfg()
    env_cfg.scene.chase = CHASE_CAMERA.replace(width=args.chase_size[0], height=args.chase_size[1])
    if getattr(env_cfg, "chase_camera_offset", None):  # indoor scenes bring their own, closer one
        pos, rot = env_cfg.chase_camera_offset
        env_cfg.scene.chase = env_cfg.scene.chase.replace(offset=TiledCameraCfg.OffsetCfg(pos=pos, rot=rot, convention="world"))
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    obs, _ = env.reset()
    robot = env.scene["robot"]
    rest = [REST_ARM_POSE[j] for j in JOINTS]

    def current_state():
        return obs["policy"]["state"][0].cpu().numpy()

    adapter = IsaacTeleopAdapter(current_state(), rest)
    try:  # the leader arm is optional: without valid calibration files, keys and gamepad still work
        leader_cal = load_calibration(os.path.join(REPO, args.leader_calibration))
        # Each sim arm uses its own follower calibration, the same one that converts recorded data.
        leader_to_sim = {side: LeaderToSim(leader_cal, ARM_CALIBRATION[side]) for side in ("left", "right")}
        print(f"[teleop] leader arm calibration {args.leader_calibration}; followers from calibration.py")
    except (OSError, ValueError, KeyError) as e:
        if args.selftest:
            raise
        leader_to_sim = None
        print(f"[teleop] leader arm input off: {e}")
    policy = None
    if args.policy:
        import xle_policies  # noqa: F401  (registers pi05_mem)
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.utils.control_utils import predict_action
        from xle_policies.pi05_mem import PI05MemPolicy

        policy = PI05MemPolicy.from_pretrained(args.policy).to("cuda").eval()
        pre, post = make_pre_post_processors(policy.config, pretrained_path=args.policy)
        task_text = env_cfg.task_description
        episode, outcomes = 1, []
        print(f"[teleop] policy drives: memory={policy.config.memory} from {args.policy}", flush=True)
    expert = noise = None
    if args.expert:
        from xle_leisaac.experts.action_noise import ActionNoise
        from xle_leisaac.experts.put_it_back_expert import PutItBackExpert

        expert = PutItBackExpert(env)
        expert.reset()
        noise = ActionNoise(env) if args.noise else None
        episode, outcomes = 1, []
        print(f"[teleop] scripted expert drives{' with recovery noise' if noise else ''}", flush=True)
    print(f"[teleop] env ready: {args.task}, action dim {env.action_manager.total_action_dim}", flush=True)

    enc = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
    dt_min = 1.0 / args.max_fps
    t_prev, fps, step = time.time(), 0.0, 0
    prev_pad_buttons = []
    driven_externally = False  # True while /action is driving the robot
    start_pose = robot.data.joint_pos[0, :3].clone()
    snap = {}
    try:
        while simulation_app.is_running():
            t0 = time.time()
            if args.selftest:
                pressed, edges = selftest_input(step)
                pad, do_reset = selftest_gamepad(step), False
                leader = {"left": selftest_leader_ticks(leader_cal)} if step >= LEADER_START else {}
                if step in (360, 400, 440, LEADER_START):  # state and base pose at each phase boundary
                    snap[step] = (current_state(), robot.data.joint_pos[0, :3].cpu().numpy().copy())
                if step >= SELFTEST_END:
                    break
            else:
                with state.lock:
                    alive = t0 - state.last_input < 1.0  # dead-man switch: browser went quiet -> stop
                    pressed = set(state.pressed) if alive else set()
                    pad = state.gamepad if alive else None
                    edges, state.edges = state.edges, set()
                    leader = dict(state.leader) if t0 - state.last_leader < LEADER_TIMEOUT else {}
                    do_reset, state.reset_requested = state.reset_requested, False
            if do_reset:
                obs, _ = env.reset()
                adapter = IsaacTeleopAdapter(current_state(), rest)
                if policy is not None:
                    policy.reset()
                if expert is not None:
                    expert.reset()

            cmd = keyboard_command(pressed, edges)
            pad_info = None
            if pad and pad.get("mapping") == "standard":
                cmd = cmd + gamepad_command(pad["axes"], pad["buttons"], prev_pad_buttons, args.deadzone)
                prev_pad_buttons = pad["buttons"]
                pad_info = pad.get("id", "gamepad")
            elif pad:
                pad_info = f"unsupported layout ({pad.get('mapping') or 'non-standard'}): {pad.get('id')}"

            with state.lock:
                real = dict(state.real_action) if t0 - state.real_action_time < REAL_ACTION_TIMEOUT else None
                source = state.real_action_source
            if expert is not None:
                action = expert.act()[0].cpu().numpy()
                if noise:
                    noise.base_enabled = expert.driving and not expert.precise
                    noise.arm_enabled = not expert.precise
                w, h = expert.cube_pixels()
                pad_info = (f"expert episode {episode}: '{expert.step_name}', cube on station {expert.station}, "
                            f"cube now {w:.0f}x{h:.0f} px at 224 ({'seen' if expert.seen else 'not yet clear'}); "
                            f"results so far: {', '.join(outcomes[-6:]) or '-'}")
            elif policy is not None:
                o = obs["policy"]
                observation = {f"observation.images.{c}": o[c][0].cpu().numpy() for c in ("head", "left_wrist", "right_wrist")}
                observation["observation.state"] = sim_to_lerobot_action(o["state"][:1])[0].astype(np.float32)
                act = predict_action(observation, policy, torch.device("cuda"), pre, post, use_amp=False, task=task_text)
                action = lerobot_to_sim_action(act.reshape(1, -1))[0].astype(np.float32)
                station = int(getattr(env, "put_it_back_station", [0])[0]) if hasattr(env, "put_it_back_station") else -1
                pad_info = (f"policy ({policy.config.memory}) episode {episode}, block on station {station}; "
                            f"results so far: {', '.join(outcomes[-6:]) or '-'}")
            elif real:
                # A real-robot action dict (e.g. from XLeRobot's Joy-Con code): fill in any key it
                # leaves out from the current state (positions) or zero (base speeds), then convert.
                current_real = dict(zip(XLEROBOT_ACTION_KEYS, sim_to_lerobot_action(current_state()[None])[0]))
                values = [real.get(k, 0.0 if k.endswith(".vel") else current_real[k]) for k in XLEROBOT_ACTION_KEYS]
                action = lerobot_to_sim_action(np.array([values]))[0].astype(np.float32)
                pad_info = f"real-robot actions from {source}"
                driven_externally = True
            else:
                if driven_externally:
                    adapter.reset(current_state())  # take over from where the external driver left the robot
                driven_externally = False
                action = adapter.step(cmd, pressed=pressed, edges=edges, gamepad=pad)
            leader_arms = []
            for name, ticks in (leader.items() if leader_to_sim else ()):  # a leader arm overrides its arm, whatever drives the rest
                try:
                    q = leader_to_sim[name]([float(t) for t in ticks][:6])
                except (KeyError, TypeError, ValueError, IndexError):
                    continue
                side = 1 if name == "right" else 0  # action 0:6 = left arm, 6:12 = right arm
                action[6 * side:6 * side + 6] = q
                adapter.set_arm(side, q)
                leader_arms.append(name)
            with torch.inference_mode():
                obs, _, terminated, truncated, _ = env.step(torch.as_tensor(action, device=env.device).unsqueeze(0))
            step += 1
            if (policy is not None or expert is not None) and (bool(terminated[0]) or bool(truncated[0])):  # env already reset
                log = env.extras.get("log", {})
                outcome = next((o for o in ("success", "wrong_station", "dropped", "time_out")
                                if log.get(f"Episode_Termination/{o}", 0.0) > 0.5), "ended")
                outcomes.append(outcome)
                print(f"[teleop] {'expert' if expert else 'policy'} episode {episode}: {outcome}", flush=True)
                episode += 1
                if policy is not None:
                    policy.reset()
                if expert is not None:
                    expert.reset()
                    if noise:
                        noise.reset()
            observation = dict(zip(XLEROBOT_ACTION_KEYS, (float(v) for v in sim_to_lerobot_action(current_state()[None])[0])))
            with state.lock:
                state.observation = observation

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
                    "leader": " + ".join(leader_arms) + " arm" if leader_arms else None,
                    "grippers_closed": adapter.gripper_closed,
                    "joints": np.round(current_state(), 3).tolist(),
                }
                state.frame_cond.notify_all()

            if args.selftest and step in (1, 59, 119, 179, 239, 299, 359, 399, 439, 479, 569):
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
        s = snap[360][0]  # keyboard phase result
        moved = snap[360][1] - start_pose.cpu().numpy()
        p0, p1, p2, p3 = (snap[k][0] for k in (360, 400, 440, LEADER_START))
        yaw = snap[400][1][2]
        drove = snap[440][1][:2] - snap[400][1][:2]
        drove_forward = drove[0] * np.cos(yaw) + drove[1] * np.sin(yaw)

        def pitch(state, side):
            return adapter.arms[side].pitch(dict(zip(JOINTS, state[6 * side:6 * side + 6])))

        final = current_state()
        want = leader_to_sim["left"](selftest_leader_ticks(leader_cal))
        leader_err = np.degrees(np.abs(final[0:6] - want))
        mapping_err = [abs(np.degrees(want[i]) - SELFTEST_LEADER_DEG[j]) for i, j in enumerate(JOINTS[:5])]
        checks = {
            "W drove forward (+x)": moved[0] > 0.1,
            "Q strafed left (+y)": moved[1] > 0.1,
            "A turned left (+yaw)": moved[2] > 0.5,
            "9/8 moved right arm (lift or elbow changed)": abs(s[7] - rest[1]) + abs(s[8] - rest[2]) > 0.1,
            "left arm stayed put (no keys for it)": abs(s[1] - rest[1]) + abs(s[2] - rest[2]) < 0.02,
            "R panned head": s[12] > 0.3,
            "V closed right gripper": s[11] < 0.2,
            "pad: right stick up moved right arm (lift/elbow)": abs(p1[7] - p0[7]) + abs(p1[8] - p0[8]) > 0.1,
            "pad: left stick moved left arm (lift/elbow)": abs(p1[1] - p0[1]) + abs(p1[2] - p0[2]) > 0.1,
            "pad: LT closed left gripper": p0[5] > 1.0 and p1[5] < 0.0,
            "pad: R3 + right stick turned right shoulder": abs(p2[6] - p1[6]) > 0.1,
            "pad: B panned head right": p2[12] < p1[12] - 0.1,
            "pad: D-pad up drove forward": drove_forward > 0.1,
            "pad: LB + left stick changed left wrist pitch": abs(pitch(p3, 0) - pitch(p2, 0)) > 0.1,
            # only meaningful when the follower uses the leader's calibration (then angles copy 1:1)
            "leader ticks -> joint angles (same calibration both sides)":
                max(mapping_err) < 0.2 if leader_cal == ARM_CALIBRATION["left"] else True,
            "leader drove left arm to its pose (< 3 deg)": leader_err.max() < 3.0,
            "right arm stayed put while the leader drove the left": np.abs(final[6:12] - p3[6:12]).max() < 0.02,
        }
        print(f"[teleop] gamepad: right arm lift/elbow {np.degrees(p1[7] - p0[7]):+.1f}/{np.degrees(p1[8] - p0[8]):+.1f} deg, "
              f"left {np.degrees(p1[1] - p0[1]):+.1f}/{np.degrees(p1[2] - p0[2]):+.1f} deg, left gripper "
              f"{np.degrees(p0[5]):.0f} -> {np.degrees(p1[5]):.0f} deg, right pan {np.degrees(p2[6] - p1[6]):+.1f} deg, "
              f"head pan {np.degrees(p2[12] - p1[12]):+.1f} deg, forward {drove_forward:.2f} m, "
              f"left pitch {np.degrees(pitch(p3, 0) - pitch(p2, 0)):+.1f} deg")
        print("[teleop] leader pose (deg): want " + " ".join(f"{np.degrees(v):.1f}" for v in want)
              + " | sim " + " ".join(f"{np.degrees(v):.1f}" for v in final[0:6]))
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
