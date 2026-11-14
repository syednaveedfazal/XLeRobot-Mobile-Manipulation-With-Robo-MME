"""joycon_sim_client.py - drive the Isaac sim on Marvin with XLeRobot's own Joy-Con teleop.

Run this on the lab laptop the Joy-Cons are paired with, in the same Python environment
you use for the real XLeRobot (lerobot + XLeRobot plugins + joycon-robotics).

It runs XLeRobot's real-robot Joy-Con code, examples/7_xlerobot_teleop_joycon.py, unchanged:
its Joy-Con classes, its SO-101 IK, its head control and its base acceleration logic. Only
the robot is swapped: instead of the real XLeRobot, a stand-in sends the same action dict
to the sim's /action endpoint and reads the sim's state from /observation. The sim converts
those real-robot values with the arm calibration, so the sim reacts like the real robot
would to the same Joy-Con input.

Usage (after `bash scripts/start_teleop_isaac.sh` printed its URL and you set up the tunnel):

    python tools/joycon_sim_client.py --url "http://localhost:22874/?token=..." \
        --xlerobot ~/XLeRobot

Open the same URL in a browser to watch the cameras while you drive.

--fake uses scripted fake Joy-Cons instead of real ones (tools/fake_joycon.py), to test the
whole chain without hardware.
"""

import argparse
import importlib.util
import json
import os
import sys
import threading
import time
import urllib.request
from urllib.parse import parse_qs, urlparse

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--url", required=True, help="the teleop URL printed by start_teleop_isaac.sh (with ?token=)")
parser.add_argument("--xlerobot", default=os.path.expanduser("~/XLeRobot"), help="path to the XLeRobot repo")
parser.add_argument("--fps", type=float, default=30.0)
parser.add_argument("--fake", action="store_true", help="scripted fake Joy-Cons (test without hardware)")
parser.add_argument("--seconds", type=float, default=0.0, help="stop after this many seconds (0 = run until Ctrl+C)")
args = parser.parse_args()

SOFTWARE = os.path.join(args.xlerobot, "software")
EXAMPLE = os.path.join(SOFTWARE, "examples", "7_xlerobot_teleop_joycon.py")


class SimLink:
    """HTTP connection to the Isaac teleop server (the same token as the browser page)."""

    def __init__(self, url):
        parsed = urlparse(url)
        self.base = f"{parsed.scheme}://{parsed.netloc}"
        self.token = parse_qs(parsed.query)["token"][0]

    def _request(self, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(f"{self.base}{path}?token={self.token}", data=data, method="POST" if data else "GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read() or b"{}")

    def observation(self):
        return self._request("/observation")

    def send(self, action):
        self._request("/action", {"action": action, "source": "XLeRobot Joy-Con code"})


class SimXLerobot:
    """Stands in for lerobot_robot_xlerobot.XLerobot in XLeRobot's Joy-Con code.

    The teleop code only uses get_observation(), send_action() and the base-keyboard helper,
    so that is all this provides. The base helper is XLeRobot's own method, called on a real
    XLerobot object that is never connected to hardware.

    Network I/O runs in a background thread: it keeps sending the newest action and fetching
    the newest observation, so the Joy-Con loop runs at its own rate even over a slow tunnel.
    """

    def __init__(self, link, xlerobot_cls, config_cls):
        self.link = link
        self._xle = xlerobot_cls(config_cls(id="sim_proxy"))  # only for its base-key logic; no connect()
        self._obs = link.observation()
        self._pending, self._outbox = {}, None
        self._lock = threading.Lock()
        self.round_trips = []  # seconds per send+fetch, for the latency report
        threading.Thread(target=self._io_loop, daemon=True).start()

    def _io_loop(self):
        while True:
            with self._lock:
                action, self._outbox = self._outbox, None
            start = time.time()
            try:
                if action:
                    self.link.send(action)
                obs = self.link.observation()
                with self._lock:
                    self._obs = obs
                self.round_trips.append(time.time() - start)
            except OSError as error:  # server busy or tunnel hiccup: keep going
                print(f"[joycon] network: {error}", flush=True)
                time.sleep(0.2)
            if not action:
                time.sleep(0.005)

    def refresh(self):
        pass  # the background thread keeps the observation fresh

    def get_observation(self):
        with self._lock:
            return dict(self._obs)

    def send_action(self, action):
        self._pending.update(action)
        return action

    def flush(self):
        if self._pending:
            with self._lock:
                self._outbox = {**(self._outbox or {}), **self._pending}  # newest values win
            self._pending = {}

    def _from_keyboard_to_base_action(self, pressed_keys):
        return self._xle._from_keyboard_to_base_action(pressed_keys)


def load_xlerobot_example():
    """Import XLeRobot's Joy-Con example as a module, unchanged."""
    for path in (os.path.join(SOFTWARE, "plugins", "lerobot_robot_xlerobot"), os.path.join(SOFTWARE, "plugins", "xlerobot_model"), SOFTWARE):
        if path not in sys.path:
            sys.path.append(path)  # used only if the packages are not pip-installed already
    if args.fake:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import fake_joycon  # noqa: F401  (registers a fake `joyconrobotics` module)
    spec = importlib.util.spec_from_file_location("xlerobot_joycon_teleop", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    xj = load_xlerobot_example()
    link = SimLink(args.url)
    robot = SimXLerobot(link, xj.XLerobot, xj.XLerobotConfig)
    print("[joycon] connected to the sim; start state:", {k: round(v, 1) for k, v in robot.get_observation().items()})

    # From here on this mirrors main() of 7_xlerobot_teleop_joycon.py, with `robot` = the sim.
    joycon_right = xj.FixedAxesJoyconRobotics("right", dof_speed=[2, 2, 2, 1, 1, 1])
    joycon_left = xj.FixedAxesJoyconRobotics("left", dof_speed=[2, 2, 2, 1, 1, 1])
    obs = robot.get_observation()
    left_arm = xj.SimpleTeleopArm(xj.LEFT_JOINT_MAP, obs, xj.SO101Kinematics(), prefix="left")
    right_arm = xj.SimpleTeleopArm(xj.RIGHT_JOINT_MAP, obs, xj.SO101Kinematics(), prefix="right")
    head_control = xj.SimpleHeadControl(obs)
    left_arm.move_to_zero_position(robot)
    right_arm.move_to_zero_position(robot)
    head_control.move_to_zero_position(robot)
    robot.flush()

    if args.fake:
        import fake_joycon

        fake_joycon.restart_clock()  # the button script starts with the control loop
    period, start = 1.0 / args.fps, time.time()
    next_print = start + 1.0
    try:
        while not args.seconds or time.time() - start < args.seconds:
            t0 = time.time()
            robot.refresh()
            pose_right, gripper_right, control_button_right = joycon_right.get_control()
            pose_left, gripper_left, control_button_left = joycon_left.get_control()

            if control_button_right == 8:  # reset button
                right_arm.move_to_zero_position(robot)
                left_arm.move_to_zero_position(robot)
                head_control.move_to_zero_position(robot)
                robot.flush()
                continue

            right_arm.target_positions["gripper"] = gripper_right
            left_arm.target_positions["gripper"] = gripper_left
            right_arm.handle_joycon_input(pose_right, gripper_right)
            right_action = right_arm.p_control_action(robot)
            left_arm.handle_joycon_input(pose_left, gripper_left)
            left_action = left_arm.p_control_action(robot)
            head_control.handle_joycon_input(joycon_left)
            head_action = head_control.p_control_action(robot)

            base_action = xj.get_joycon_base_action(joycon_right, robot)
            speed_multiplier = xj.get_joycon_speed_control(joycon_right)
            if base_action:
                for key in base_action:
                    if "vel" in key or "velocity" in key:
                        base_action[key] *= speed_multiplier

            robot.send_action({**left_action, **right_action, **head_action, **base_action})
            robot.flush()
            if args.fake and time.time() >= next_print:  # once a second in tests
                next_print += 1.0
                o = robot.get_observation()
                print(f"[joycon] t={time.time() - start:4.1f}s  x.vel={o['x.vel']:+.2f}  "
                      f"R lift/elbow={o['right_arm_shoulder_lift.pos']:+6.1f}/{o['right_arm_elbow_flex.pos']:+6.1f}  "
                      f"R grip={o['right_arm_gripper.pos']:5.1f}  head1={o['head_motor_1.pos']:+6.1f}", flush=True)
            time.sleep(max(0.0, period - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        joycon_right.disconnect()
        joycon_left.disconnect()
    time.sleep(0.5)
    trips = sorted(robot.round_trips) or [0.0]
    print(f"[joycon] network round trip (send+fetch): median {trips[len(trips) // 2] * 1000:.0f} ms, "
          f"95% {trips[int(len(trips) * 0.95)] * 1000:.0f} ms over {len(trips)} exchanges")
    print("[joycon] end state:", {k: round(v, 1) for k, v in robot.get_observation().items()})


if __name__ == "__main__":
    main()
