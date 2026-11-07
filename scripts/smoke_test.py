"""Headless check that XLeRobot loads, renders and responds to keyboard + gamepad teleop.

    python scripts/smoke_test.py            # writes outputs/smoke_test_*.png

Feeds scripted inputs through the same TeleopController / input mappings the web teleop
uses and asserts that the robot moves in the expected directions.
"""

from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np

import xlerobot_sim
from xlerobot_sim.teleop_controller import (
    TeleopCommand, TeleopController, gamepad_command, get_mapped_joints, keyboard_command,
)
from xlerobot_sim.teleop_web import to_uint8_image

OUT = Path(__file__).resolve().parent.parent / "outputs"


def pad(axes=(0, 0, 0, 0), **buttons):
    """Standard-mapping gamepad state; buttons by name, e.g. pad(axes=(0,0,0,-1), RT=1.0)."""
    names = ["A", "B", "X", "Y", "LB", "RB", "LT", "RT", "VIEW", "MENU", "L3", "R3", "UP", "DOWN", "LEFT", "RIGHT"]
    return list(axes), [float(buttons.get(n, 0.0)) for n in names] + [0.0]


def main():
    env = gym.make(
        xlerobot_sim.ENV_ID,
        obs_mode="rgb",
        control_mode="pd_joint_delta_pos_dual_arm",
        render_mode="rgb_array",
        sensor_configs=dict(shader_pack="default"),
        human_render_camera_configs=dict(shader_pack="default"),
    )
    obs, _ = env.reset(seed=0, options=dict(reconfigure=True))
    agent = env.unwrapped.agent
    robot = agent.robot
    print("action space:", env.action_space.shape, "| cameras:", list(obs["sensor_data"].keys()))

    ctrl = TeleopController()
    ctrl.reset(get_mapped_joints(robot))

    def run(cmd_fn, n):
        nonlocal obs
        for _ in range(n):
            obs, *_ = env.step(ctrl.step(cmd_fn(), get_mapped_joints(robot)))

    def tcp(arm):  # world position of the fixed jaw of arm 1 (right) / arm 2 (left)
        return (agent.tcp if arm == 1 else agent.tcp_2).pose.p[0].cpu().numpy().copy()

    run(TeleopCommand, ctrl.warmup_steps + 30)  # warmup + settle
    checks = {}

    # --- gamepad: right stick up -> arm 1 (right) EE up; left stick up -> arm 2 (left) EE up
    z1, z2 = tcp(1)[2], tcp(2)[2]
    run(lambda: gamepad_command(*pad(axes=(0, -1, 0, -1)), []), 40)
    run(TeleopCommand, 15)
    checks["pad right stick up raises arm 1"] = tcp(1)[2] - z1
    checks["pad left stick up raises arm 2"] = tcp(2)[2] - z2

    # --- gamepad: R3 + right stick right -> arm 1 swings to the robot's right (-y, robot faces +x)
    y1 = tcp(1)[1]
    run(lambda: gamepad_command(*pad(axes=(0, 0, 1, 0), R3=1), []), 40)
    run(TeleopCommand, 15)
    checks["pad R3+right stick right swings arm 1 right (-y)"] = y1 - tcp(1)[1]
    y1 = tcp(1)[1]
    run(lambda: gamepad_command(*pad(axes=(0, 0, -1, 0), R3=1), []), 40)  # swing back
    run(TeleopCommand, 15)

    # --- gamepad: RT closes gripper 1 (analog), releasing opens it again
    g_open = get_mapped_joints(robot)[12]
    run(lambda: gamepad_command(*pad(RT=1.0), []), 80)
    g_closed = get_mapped_joints(robot)[12]
    checks["pad RT closes gripper 1"] = g_open - g_closed

    # --- keyboard: W drives forward (+x), A turns left (+yaw)
    j0 = get_mapped_joints(robot)
    run(lambda: keyboard_command({"w"}, set()), 60)
    run(lambda: keyboard_command({"a"}, set()), 40)
    j1 = get_mapped_joints(robot)
    checks["key W drives forward"] = j1[0] - j0[0]
    checks["key A turns left"] = j1[1] - j0[1]

    # --- gamepad: D-pad up drives forward
    x0 = get_mapped_joints(robot)[0]
    run(lambda: gamepad_command(*pad(UP=1), []), 40)
    checks["pad D-pad up drives forward"] = get_mapped_joints(robot)[0] - x0

    OUT.mkdir(exist_ok=True)
    cv2.imwrite(str(OUT / "smoke_test_main.png"), cv2.cvtColor(to_uint8_image(env.render()), cv2.COLOR_RGB2BGR))
    for uid, cam in obs["sensor_data"].items():
        cv2.imwrite(str(OUT / f"smoke_test_{uid}.png"), cv2.cvtColor(to_uint8_image(cam["rgb"]), cv2.COLOR_RGB2BGR))
    env.close()

    failed = [name for name, v in checks.items() if not v > 0.01]
    for name, v in checks.items():
        print(f"  [{'ok' if name not in failed else 'FAIL'}] {name}: {v:+.3f}")
    print("images written to", OUT)
    assert not failed, f"failed: {failed}"
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
