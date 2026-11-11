"""Browser-based teleop (keyboard + gamepad) for XLeRobot in ManiSkill (headless-friendly).

The simulator renders off-screen on the GPU node; camera frames are streamed to
your browser as MJPEG. Keyboard and gamepad input are read by the browser (Gamepad
API -- the controller is plugged into *your laptop*) and sent back over HTTP.
Nothing needs a local display on the cluster, so this works from a plain Slurm job.

    python -m xlerobot_sim.teleop_web --port 8765

then open the printed URL through an SSH tunnel / relay (see README).
"""

from __future__ import annotations

import json
import os
import secrets
import socket
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import tyro

from xlerobot_sim.teleop_controller import (
    CONTROL_MODE,
    GAMEPAD_HELP,
    KEY_HELP,
    TeleopController,
    gamepad_command,
    get_mapped_joints,
    keyboard_command,
)


@dataclass
class Args:
    env_id: str = "XLeRobot-ReplicaCAD-v1"
    """ManiSkill environment id."""
    port: Optional[int] = None
    """HTTP port of the teleop page. Default: 20000 + uid % 20000 (per-user, avoids clashes on shared
    nodes). If taken, the next free port is used."""
    bind: str = "0.0.0.0"
    """Interface to bind. 0.0.0.0 is needed to reach a compute node from the login node."""
    shader: str = "default"
    """Shader pack for cameras: 'minimal' (fastest), 'default', 'rt-fast' (ray traced, slow)."""
    sim_backend: str = "auto"
    """'auto', 'cpu' or 'gpu'."""
    seed: Optional[int] = 0
    """Seed for the scene."""
    max_fps: float = 30.0
    """Upper bound on control/render rate."""
    jpeg_quality: int = 80
    deadzone: float = 0.15
    """Gamepad stick deadzone."""
    token: Optional[str] = None
    """Access token required in the URL. Random if not given."""


def to_uint8_image(x) -> np.ndarray:
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    x = np.asarray(x)
    if x.ndim == 4:
        x = x[0]
    if x.dtype != np.uint8:
        x = (np.clip(x, 0, 1) * 255).astype(np.uint8) if x.max() <= 1.0 else x.astype(np.uint8)
    return x[..., :3]


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame_cond = threading.Condition(self.lock)
        self.frames: dict[str, bytes] = {}
        self.frame_id = 0
        self.pressed: set[str] = set()
        self.edges: set[str] = set()
        self.gamepad: Optional[dict] = None  # {"id", "mapping", "axes", "buttons"}
        self.last_input = 0.0
        self.reset_requested = False
        self.status: dict = {}
        self.running = True


def make_handler(state: SharedState, token: str, page: str):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # keep-alive: input is posted ~30x/s through the tunnel
        disable_nagle_algorithm = True  # otherwise each small response waits ~40 ms (Nagle + delayed ACK)

        def log_message(self, *a):  # keep stdout clean
            pass

        def _authorized(self) -> bool:
            q = parse_qs(urlparse(self.path).query)
            if q.get("token", [None])[0] == token:
                return True
            self.send_error(403, "missing or wrong ?token=")
            return False

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self._authorized():
                return
            path = urlparse(self.path).path
            if path == "/":
                self._send(200, page.encode(), "text/html; charset=utf-8")
            elif path == "/status":
                with state.lock:
                    body = json.dumps(state.status).encode()
                self._send(200, body, "application/json")
            elif path.startswith("/stream/"):
                self._stream(path.split("/")[-1])
            else:
                self.send_error(404)

        def _stream(self, cam: str):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")  # endless body: no keep-alive for streams
            self.end_headers()
            self.close_connection = True
            last = -1
            try:
                while state.running:
                    with state.frame_cond:
                        state.frame_cond.wait_for(lambda: state.frame_id != last or not state.running, timeout=1.0)
                        last = state.frame_id
                        jpg = state.frames.get(cam)
                    if jpg is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            if not self._authorized():
                return
            path = urlparse(self.path).path
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
            with state.lock:
                if path == "/input":
                    state.pressed = {str(k).lower() for k in data.get("pressed", [])}
                    state.edges |= {str(k).lower() for k in data.get("down", [])}
                    state.gamepad = data.get("gamepad")
                    state.last_input = time.time()
                elif path == "/reset":
                    state.reset_requested = True
            self._send(200, b"{}", "application/json")

    return Handler


def _help_rows(groups) -> str:
    rows = []
    for group, items in groups:
        rows.append(f"<tr><th colspan=2>{group}</th></tr>")
        rows += [f"<tr><td><kbd>{k}</kbd></td><td>{d}</td></tr>" for k, d in items]
    return "\n".join(rows)


def build_page(token: str) -> str:
    return (PAGE_TEMPLATE.replace("__TOKEN__", token)
            .replace("__KEYS__", _help_rows(KEY_HELP))
            .replace("__PAD__", _help_rows(GAMEPAD_HELP)))


def main(args: Args):
    import gymnasium as gym

    import xlerobot_sim  # noqa: F401  registers the xlerobot agent and env

    token = args.token or secrets.token_urlsafe(12)
    state = SharedState()
    handler = make_handler(state, token, build_page(token))
    first = args.port or 20000 + os.getuid() % 20000
    for port in range(first, first + 20):
        try:
            server = ThreadingHTTPServer((args.bind, port), handler)
            break
        except OSError:
            continue
    else:
        raise SystemExit(f"no free port in {first}..{first + 19}")
    args.port = port
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()

    host = socket.gethostname().split(".")[0]
    print("=" * 72)
    print(f" Teleop server on {host}:{args.port}")
    print(f" Open: http://localhost:{args.port}/?token={token}")
    print("=" * 72, flush=True)

    env = gym.make(
        args.env_id,
        robot_uids="xlerobot",
        obs_mode="rgb",
        control_mode=CONTROL_MODE,
        render_mode="rgb_array",
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        sim_backend=args.sim_backend,
        enable_shadow=True,
        num_envs=1,
    )
    obs, _ = env.reset(seed=args.seed, options=dict(reconfigure=True))
    agent = env.unwrapped.agent
    bounds = env.unwrapped.single_action_space
    ctrl = TeleopController(bounds.low, bounds.high)
    ctrl.reset(get_mapped_joints(agent))
    print(f"[teleop] env ready: {args.env_id}, action dim {env.action_space.shape}", flush=True)

    dt_min = 1.0 / args.max_fps
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
    t_prev, fps = time.time(), 0.0
    prev_pad_buttons: list[float] = []
    try:
        while True:
            t0 = time.time()
            with state.lock:
                alive = t0 - state.last_input < 1.0  # dead-man switch: browser went quiet -> stop
                pressed = set(state.pressed) if alive else set()
                pad = state.gamepad if alive else None
                edges, state.edges = state.edges, set()
                do_reset, state.reset_requested = state.reset_requested, False
            if do_reset:
                obs, _ = env.reset(options=dict(reconfigure=False))
                ctrl = TeleopController(bounds.low, bounds.high)
                ctrl.reset(get_mapped_joints(agent))

            cmd = keyboard_command(pressed, edges)
            pad_info = None
            if pad and pad.get("mapping") == "standard":
                cmd = cmd + gamepad_command(pad["axes"], pad["buttons"], prev_pad_buttons, args.deadzone)
                prev_pad_buttons = pad["buttons"]
                pad_info = pad.get("id", "gamepad")
            elif pad:
                pad_info = f"unsupported layout ({pad.get('mapping') or 'non-standard'}): {pad.get('id')}"

            current = get_mapped_joints(agent)
            action = ctrl.step(cmd, current)
            obs, *_ = env.step(action)

            frames = {"main": to_uint8_image(env.render())}
            sensors = obs.get("sensor_data", {}) if isinstance(obs, dict) else {}
            for cam_uid, name in (("fetch_head", "head"), ("fetch_right_arm_camera", "wrist1"),
                                  ("fetch_left_arm_camera", "wrist2")):
                if cam_uid in sensors and "rgb" in sensors[cam_uid]:
                    frames[name] = to_uint8_image(sensors[cam_uid]["rgb"])
            jpgs = {k: cv2.imencode(".jpg", cv2.cvtColor(v, cv2.COLOR_RGB2BGR), enc)[1].tobytes()
                    for k, v in frames.items()}

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now
            with state.frame_cond:
                state.frames = jpgs
                state.frame_id += 1
                state.status = {
                    "fps": round(fps, 1),
                    "step": ctrl.step_counter,
                    "warmup": ctrl.warming_up,
                    "pressed": sorted(pressed),
                    "gamepad": pad_info,
                    "grippers_closed": ctrl.gripper_closed,
                    "joints": np.round(current, 3).tolist(),
                }
                state.frame_cond.notify_all()

            sleep = dt_min - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    finally:
        state.running = False
        server.shutdown()
        env.close()


PAGE_TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>XLeRobot Teleop</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 :root { --bg:#14161a; --fg:#e6e6e6; --mut:#8b9099; --acc:#e0893a; --card:#1d2026; }
 body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.4 system-ui, sans-serif; }
 header { padding:10px 16px; display:flex; flex-wrap:wrap; gap:12px; align-items:center; border-bottom:1px solid #2a2e36; }
 header b { color:var(--acc); }
 .pill { padding:2px 8px; border-radius:4px; background:#5a2a2a; }
 .pill.on { background:#2a5a34; }
 main { display:grid; grid-template-columns: minmax(0,3fr) minmax(280px,1fr); gap:12px; padding:12px 16px; }
 .cams img { width:100%; background:#000; border-radius:6px; display:block; }
 .small { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin-top:8px; }
 .small figure { margin:0; } figcaption { color:var(--mut); font-size:12px; }
 .tabs button.sel { border-color:var(--acc); color:var(--acc); }
 table { border-collapse:collapse; width:100%; background:var(--card); border-radius:6px; margin-top:8px; }
 th { text-align:left; color:var(--acc); padding:6px 8px 2px; }
 td { padding:2px 8px; } kbd { background:#2c313a; padding:1px 5px; border-radius:3px; white-space:nowrap; }
 button { background:#2c313a; color:var(--fg); border:1px solid #3a404a; border-radius:4px; padding:4px 10px; cursor:pointer; }
 #status, #raw { color:var(--mut); font-family:monospace; font-size:12px; white-space:pre-wrap; margin-top:8px; }
 @media (max-width: 800px) { main { grid-template-columns: 1fr; } }
</style></head><body>
<header><b>XLeRobot teleop</b>
 <span id="focus" class="pill">click the page to capture keys</span>
 <span id="pad" class="pill">no gamepad: press a button on it</span>
 <button id="reset">Reset episode</button><span id="fps"></span></header>
<main>
 <div class="cams">
  <img src="stream/main?token=__TOKEN__" alt="third-person view">
  <div class="small">
   <figure><img src="stream/head?token=__TOKEN__"><figcaption>head camera</figcaption></figure>
   <figure><img src="stream/wrist1?token=__TOKEN__"><figcaption>arm 1 (right) wrist</figcaption></figure>
   <figure><img src="stream/wrist2?token=__TOKEN__"><figcaption>arm 2 (left) wrist</figcaption></figure>
  </div>
 </div>
 <div>
  <div class="tabs"><button id="t-pad">Gamepad</button> <button id="t-key">Keyboard</button>
   <label style="margin-left:8px;color:var(--mut)"><input type="checkbox" id="showraw"> raw gamepad</label></div>
  <table id="h-pad">__PAD__</table>
  <table id="h-key">__KEYS__</table>
  <div id="raw"></div>
  <div id="status"></div>
 </div>
</main>
<script>
const T = "token=__TOKEN__";
const pressed = new Set();
let down = [];

function readPad() {
  const pads = navigator.getGamepads ? [...navigator.getGamepads()].filter(Boolean) : [];
  const p = pads.find(p => p.mapping === "standard") || pads[0];
  if (!p) return null;
  return {id: p.id, mapping: p.mapping, axes: [...p.axes], buttons: p.buttons.map(b => b.value)};
}
function send() {
  const gamepad = readPad();
  fetch("input?" + T, {method:"POST", body: JSON.stringify({pressed:[...pressed], down, gamepad})}).catch(()=>{});
  down = [];
  showPad(gamepad);
}
// ~30 Hz while a gamepad is connected, 5 Hz heartbeat otherwise (server stops the robot if this goes quiet)
let last = 0;
setInterval(() => { const now = performance.now();
  if (readPad() || now - last > 200) { last = now; send(); } }, 33);

function norm(e) { return e.key.length === 1 ? e.key.toLowerCase() : e.key; }
addEventListener("keydown", e => {
  if (e.key === "Escape") { pressed.clear(); send(); return; }
  if (e.ctrlKey || e.metaKey) return;
  const k = norm(e); e.preventDefault();
  if (!pressed.has(k)) { pressed.add(k); down.push(k); send(); }
});
addEventListener("keyup", e => { pressed.delete(norm(e)); send(); });
addEventListener("blur", () => { pressed.clear(); send(); focusPill(false); });
addEventListener("focus", () => focusPill(true));
function focusPill(on) { const f = document.getElementById("focus");
  f.className = "pill" + (on ? " on" : ""); f.textContent = on ? "keys captured" : "click the page to capture keys"; }
focusPill(document.hasFocus());

function showPad(g) {
  const el = document.getElementById("pad");
  if (!g) { el.className = "pill"; el.textContent = "no gamepad: press a button on it"; }
  else if (g.mapping !== "standard") { el.className = "pill"; el.textContent = "unsupported layout: " + g.id; }
  else { el.className = "pill on"; el.textContent = "gamepad: " + g.id.slice(0, 40); }
  const raw = document.getElementById("raw");
  raw.textContent = (document.getElementById("showraw").checked && g)
    ? `${g.id}\nmapping: ${g.mapping || "(none)"}\naxes: ${g.axes.map(a=>a.toFixed(2)).join(" ")}\n` +
      "buttons: " + g.buttons.map((b,i)=>`${i}:${b.toFixed(1)}`).join(" ") : "";
}
function tab(which) {
  for (const w of ["pad","key"]) {
    document.getElementById("h-"+w).style.display = w === which ? "" : "none";
    document.getElementById("t-"+w).className = w === which ? "sel" : "";
  }
}
document.getElementById("t-pad").onclick = () => tab("pad");
document.getElementById("t-key").onclick = () => tab("key");
tab("pad");
addEventListener("gamepadconnected", () => tab("pad"));

document.getElementById("reset").onclick = () => fetch("reset?" + T, {method:"POST", body:"{}"});
setInterval(async () => {
  try { const s = await (await fetch("status?" + T)).json();
    document.getElementById("fps").textContent = `${s.fps} Hz` + (s.warmup ? "  (warming up)" : "");
    document.getElementById("status").textContent =
      "keys: " + (s.pressed.join(" ") || "-") +
      "\ngrippers closed (R, L): " + s.grippers_closed.join(", ") +
      "\njoints: " + s.joints.map(x=>x.toFixed(2)).join(" ");
  } catch (e) {}
}, 500);
</script></body></html>
"""

if __name__ == "__main__":
    main(tyro.cli(Args))
