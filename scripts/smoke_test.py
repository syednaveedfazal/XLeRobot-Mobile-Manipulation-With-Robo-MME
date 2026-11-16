"""Headless check that XLeRobot loads, renders and responds to keyboard + gamepad teleop.

    python scripts/smoke_test.py            # writes outputs/smoke_test_*.png

Feeds scripted inputs through the same TeleopController / input mappings the web teleop
uses and asserts that the robot moves in the expected directions. Also checks that the model
matches the real robot where we fixed it: 17-dim action, ~12 kg, 640x480 cameras, base speed clamp.
"""

from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import torch

import xlerobot_sim
from xlerobot_sim import layout as L
from xlerobot_sim.teleop_controller import (
    CONTROL_MODE, TeleopCommand, TeleopController, gamepad_command, get_mapped_joints, keyboard_command,
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
        control_mode=CONTROL_MODE,
        render_mode="rgb_array",
        sensor_configs=dict(shader_pack="default"),
        human_render_camera_configs=dict(shader_pack="default"),
    )
    obs, _ = env.reset(seed=0, options=dict(reconfigure=True))
    agent = env.unwrapped.agent
    robot = agent.robot
    bounds = env.unwrapped.single_action_space
    print("action space:", bounds.shape, "| cameras:", list(obs["sensor_data"].keys()))
    checks = {}

    # --- model matches the real robot where we fixed it
    mass = sum(link._objs[0].mass for link in robot.get_links())
    print(f"  total mass {mass:.2f} kg | state_real_layout {tuple(obs['agent']['state_real_layout'].shape)}")
    checks["action is 17-dim (real robot layout)"] = float(bounds.shape == (L.ACTION_DIM,))
    checks["state_real_layout is 17-dim"] = float(obs["agent"]["state_real_layout"].shape[-1] == L.ACTION_DIM)
    checks["total mass within 11..13.5 kg"] = float(11.0 < mass < 13.5)
    sizes = {uid: tuple(cam["rgb"].shape[-3:-1]) for uid, cam in obs["sensor_data"].items()}
    print("  camera (h, w):", sizes)
    checks["cameras are 640x480"] = float(len(sizes) == 3 and all(hw == (480, 640) for hw in sizes.values()))

    ctrl = TeleopController(bounds.low, bounds.high)
    ctrl.reset(get_mapped_joints(agent))

    def run(cmd_fn, n):
        nonlocal obs
        for _ in range(n):
            obs, *_ = env.step(ctrl.step(cmd_fn(), get_mapped_joints(agent)))

    def base():  # x, y (m), yaw (rad)
        return agent.base_pose()[0].cpu().numpy().copy()

    def tcp(arm):  # world position of the fixed jaw of arm 1 (right) / arm 2 (left)
        return (agent.tcp if arm == 1 else agent.tcp_2).pose.p[0].cpu().numpy().copy()

    run(TeleopCommand, ctrl.warmup_steps + 30)  # warmup + settle

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
    g_open = get_mapped_joints(agent)[L.RIGHT_GRIPPER]
    run(lambda: gamepad_command(*pad(RT=1.0), []), 80)
    g_closed = get_mapped_joints(agent)[L.RIGHT_GRIPPER]
    checks["pad RT closes gripper 1"] = g_open - g_closed
    run(TeleopCommand, 40)  # release: gripper opens again

    # --- keyboard: W drives forward (+x), A turns left (+yaw)
    b0 = base()
    run(lambda: keyboard_command({"w"}, set()), 60)
    run(lambda: keyboard_command({"a"}, set()), 40)
    b1 = base()
    checks["key W drives forward"] = b1[0] - b0[0]
    checks["key A turns left"] = b1[2] - b0[2]

    # --- keyboard: Q slides left (+y in the robot frame), without turning
    b0 = base()
    run(lambda: keyboard_command({"q"}, set()), 60)
    b1 = base()
    d, c, s = b1[:2] - b0[:2], np.cos(b0[2]), np.sin(b0[2])
    checks["key Q slides left (robot frame)"] = -s * d[0] + c * d[1]
    checks["key Q does not turn"] = float(abs(b1[2] - b0[2]) < 0.02)

    # --- gamepad: D-pad up drives forward
    b0 = base()
    run(lambda: gamepad_command(*pad(UP=1), []), 40)
    b1 = base()
    checks["pad D-pad up drives forward"] = np.cos(b0[2]) * (b1[0] - b0[0]) + np.sin(b0[2]) * (b1[1] - b0[1])

    # --- base commands are clamped to the real robot's fastest preset (0.3 m/s, 90 deg/s).
    # Checked on the controller: a physical test depends on where the robot ended up in the room.
    base_ctrl = agent.controller.controllers["base"]
    clamped = base_ctrl._preprocess_action(torch.tensor([[5.0, -5.0, 9.0]], device=agent.device))[0].cpu().numpy()
    print("  base command (5, -5, 9) ->", np.round(clamped, 4))
    checks["base command clamped to 0.3 m/s, 90 deg/s"] = float(
        np.allclose(clamped, [L.BASE_MAX_XY, -L.BASE_MAX_XY, L.BASE_MAX_YAW]))

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
