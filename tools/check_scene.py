"""check_scene.py - load a scene env, check its physics, and take pictures of it.

  sbatch --output=$HOME/xle_scene_%j.out slurm/sim_job.sbatch \
      /isaac-sim/python.sh tools/check_scene.py --enable_cameras --device cpu

Checks (prints PASS/FAIL per line, then SCENE CHECK PASSED/FAILED):
1. settle   3 s of holding still: no object falls through the floor, few objects move,
            furniture joints stay shut, the robot stays at its start.
2. furniture every drawer and door is set 40 % open; at most a quarter of them may then be
            pushed back by something in their way (1 s).
3. drive    1 s forward at 0.3 m/s: the robot moves about 0.3 m.

Pictures go to $XLE_LOGS/scene_check/<task>/: survey views of the rooms (drawers shut and
open), and the robot's head and wrist cameras.
"""

import argparse
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="XLeRobot-ReplicaCAD-v0")
parser.add_argument("--settle_s", type=float, default=3.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import xle_leisaac  # noqa: F401, E402
from xle_leisaac.robots.xlerobot import BASE_JOINTS, HEAD_JOINTS, LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS  # noqa: E402

# Survey views (eye, look-at) in world metres, for ReplicaCAD apt_0. The kitchen is along the
# x = -2.5 wall, the living room at y < -4, the robot starts at (-1, 0).
VIEWS = {
    "kitchen": ((-0.7, 0.5, 1.8), (-2.3, -1.6, 0.9)),
    "robot": ((0.4, -1.4, 1.5), (-1.0, 0.0, 0.5)),
    "chest_of_drawers": ((-0.3, 2.6, 1.4), (-2.3, 1.3, 0.6)),
    "fridge": ((-0.4, -1.2, 1.5), (-2.3, -3.2, 0.9)),
    "living_room": ((-0.5, -2.0, 1.7), (2.5, -6.0, 0.6)),
    "cabinet": ((-0.2, -0.2, 1.4), (1.3, -1.9, 0.5)),
}

results = []


def check(name, ok, detail):
    results.append(ok)
    print(f"[scene] {'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)


def main():
    out_dir = os.path.join(os.environ["XLE_LOGS"], "scene_check", args.task)
    os.makedirs(out_dir, exist_ok=True)

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.terminations.time_out = None
    env_cfg.scene.survey = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/survey",
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=14.0, clipping_range=(0.05, 50.0)),
        width=1280,
        height=720,
    )
    t0 = time.time()
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    obs, _ = env.reset()
    print(f"[scene] env ready in {time.time() - t0:.0f} s", flush=True)

    robot = env.scene["robot"]
    objects = env.scene.rigid_object_collections.get("objects")
    furniture = {name: art for name, art in env.scene.articulations.items() if name != "robot"}
    survey = env.scene["survey"]
    dt = env.step_dt

    hold_ids = robot.find_joints(LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + HEAD_JOINTS, preserve_order=True)[0]
    base_ids = robot.find_joints(BASE_JOINTS, preserve_order=True)[0]
    hold = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
    hold[:, :14] = robot.data.default_joint_pos[:, hold_ids]

    def run(seconds, action=hold):
        nonlocal obs
        for _ in range(max(1, round(seconds / dt))):
            obs, *_ = env.step(action)

    def pictures(tag):
        for view, (eye, target) in VIEWS.items():
            origin = env.scene.env_origins[0]
            survey.set_world_poses_from_view(
                torch.tensor([eye], device=env.device) + origin, torch.tensor([target], device=env.device) + origin
            )
            run(6 * dt)  # let the renderer converge
            imageio.imwrite(os.path.join(out_dir, f"{tag}_{view}.png"), survey.data.output["rgb"][0].cpu().numpy())
        for cam in ("head", "left_wrist", "right_wrist"):
            imageio.imwrite(os.path.join(out_dir, f"{tag}_robot_{cam}.png"), obs["policy"][cam][0].cpu().numpy())

    # 1. settle
    t0 = time.time()
    run(args.settle_s)
    rate = args.settle_s / dt / (time.time() - t0)
    print(f"[scene] {rate:.1f} env steps/s with cameras ({1 / dt:.0f} needed for real time)", flush=True)
    if objects is not None:
        start = objects.data.default_object_state[0, :, :3] + env.scene.env_origins[0]
        now = objects.data.object_pos_w[0]
        moved = torch.linalg.norm(now - start, dim=-1)
        names = objects.object_names
        fell = [names[i] for i in torch.nonzero(now[:, 2] < -0.1).flatten().tolist()]
        movers = sorted(((moved[i].item(), names[i]) for i in torch.nonzero(moved > 0.03).flatten().tolist()), reverse=True)
        check("objects stay in the apartment", not fell and torch.isfinite(now).all().item(), f"fell through floor: {fell or 'none'}")
        check(
            "objects settle",
            len(movers) <= max(3, len(names) // 20),
            f"{len(movers)}/{len(names)} moved > 3 cm; largest: "
            + ", ".join(f"{n} {d * 100:.0f} cm" for d, n in movers[:5]),
        )
    for name, art in furniture.items():
        drift = art.data.joint_pos.abs().max().item()
        check(f"{name} stays shut", drift < 0.01, f"max joint drift {drift:.4f}")
    base_drift = robot.data.joint_pos[0, base_ids].abs().max().item()
    check("robot holds its start pose", base_drift < 0.02, f"base joint drift {base_drift:.3f}")
    pictures("closed")

    # 2. furniture: open everything to 40 %, it should stay open
    targets = {}
    for name, art in furniture.items():
        lower, upper = art.data.joint_pos_limits[0, :, 0], art.data.joint_pos_limits[0, :, 1]
        target = torch.where(upper.abs() >= lower.abs(), 0.4 * upper, 0.4 * lower).unsqueeze(0)
        art.write_joint_state_to_sim(target, torch.zeros_like(target))
        targets[name] = target
    run(1.0)
    for name, art in furniture.items():
        print(f"[scene] {name}: set {[round(v, 3) for v in targets[name][0].tolist()]} "
              f"-> after 1 s {[round(v, 3) for v in art.data.joint_pos[0].tolist()]}", flush=True)
    pushed, total = [], 0
    for name, art in furniture.items():
        change = (art.data.joint_pos - targets[name]).abs()[0]
        total += len(change)
        pushed += [f"{name}/{art.joint_names[j]} {change[j]:.3f}" for j in torch.nonzero(change > 0.03).flatten().tolist()]
    check("drawers and doors stay open", len(pushed) <= total // 4, f"{len(pushed)}/{total} moved: {', '.join(pushed) or 'none'}")
    pictures("open")

    # 3. drive forward
    x0 = robot.data.joint_pos[0, base_ids[0]].item()
    drive = hold.clone()
    drive[:, 14] = 0.3  # forward m/s
    run(1.0, drive)
    print(f"[scene] drive: while moving, base joint vel {robot.data.joint_vel[0, base_ids].tolist()}", flush=True)
    run(0.3)
    moved = robot.data.joint_pos[0, base_ids[0]].item() - x0
    print(f"[scene] drive: base joint vel {robot.data.joint_vel[0, base_ids].tolist()}, "
          f"target {robot.data.joint_vel_target[0, base_ids].tolist()}, "
          f"x/y/yaw {robot.data.joint_pos[0, base_ids].tolist()}", flush=True)
    check("robot drives forward", 0.2 < moved < 0.45, f"moved {moved:.3f} m for 0.3 m commanded")
    imageio.imwrite(os.path.join(out_dir, "after_drive_robot_head.png"), obs["policy"]["head"][0].cpu().numpy())

    print(f"[scene] pictures in {out_dir}", flush=True)
    print(f"[scene] SCENE CHECK {'PASSED' if all(results) else 'FAILED'}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
