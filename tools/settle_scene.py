"""settle_scene.py - let a scene's objects come to rest once, and make that their start pose.

Scene poses converted from another simulator (ReplicaCAD comes from Habitat/Bullet) are not quite
at rest in PhysX: knives slide out of their block, books tip over. Without this, every reset
replays that settling at the start of an episode. This runs the scene for a few seconds with the
robot holding still and writes the final object poses into the manifest. The original poses are
kept as "pos_source"/"rot_source", so running it again starts from them, not from the last result.

It also prints the env's speed with only the robot's cameras (what teleop and recording render).

  sbatch --output=$HOME/xle_settle_%j.out slurm/sim_job.sbatch \
      /isaac-sim/python.sh tools/settle_scene.py --enable_cameras --device cpu
"""

import argparse
import json
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="XLeRobot-ReplicaCAD-v0")
parser.add_argument("--seconds", type=float, default=6.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import xle_leisaac  # noqa: F401, E402
from xle_leisaac.robots.xlerobot import HEAD_JOINTS, LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS  # noqa: E402


def main():
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.terminations.time_out = None
    manifest_path = os.path.join(os.environ["XLE_ASSETS"], env_cfg.scene_manifest)
    with open(manifest_path) as f:
        manifest = json.load(f)
    # Always settle from the converter's poses, not from a previous settle.
    for o in manifest["objects"]:
        o.setdefault("pos_source", list(o["pos"]))
        o.setdefault("rot_source", list(o["rot"]))
    for name, cfg in env_cfg.scene.objects.rigid_objects.items():
        source = next(o for o in manifest["objects"] if o["name"] == name)
        cfg.init_state.pos, cfg.init_state.rot = tuple(source["pos_source"]), tuple(source["rot_source"])

    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.reset()
    robot, objects = env.scene["robot"], env.scene["objects"]
    hold_ids = robot.find_joints(LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + HEAD_JOINTS, preserve_order=True)[0]
    hold = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
    hold[:, :14] = robot.data.default_joint_pos[:, hold_ids]

    steps = round(args.seconds / env.step_dt)
    t0 = time.time()
    for _ in range(steps):
        env.step(hold)
    rate = steps / (time.time() - t0)
    print(f"[settle] {rate:.1f} env steps/s with the robot's cameras ({1 / env.step_dt:.0f} = real time)", flush=True)

    start = objects.data.default_object_state[0, :, :3] + env.scene.env_origins[0]
    pos = objects.data.object_pos_w[0] - env.scene.env_origins[0]
    quat = objects.data.object_quat_w[0]
    speed = torch.linalg.norm(objects.data.object_lin_vel_w[0], dim=-1)
    moved = torch.linalg.norm(objects.data.object_pos_w[0] - start, dim=-1)
    names = objects.object_names
    for i in torch.argsort(moved, descending=True).tolist():
        if moved[i] > 0.03:
            print(f"[settle] {names[i]} moved {moved[i] * 100:.0f} cm, still moving {speed[i]:.3f} m/s", flush=True)
    restless = [names[i] for i in torch.nonzero(speed > 0.02).flatten().tolist()]
    fell = [names[i] for i in torch.nonzero(pos[:, 2] < -0.1).flatten().tolist()]
    if fell:
        raise SystemExit(f"[settle] objects fell through the floor: {fell}; manifest not changed")

    by_name = {o["name"]: o for o in manifest["objects"]}
    for i, name in enumerate(names):
        by_name[name]["pos"] = [round(v, 5) for v in pos[i].tolist()]
        by_name[name]["rot"] = [round(v, 6) for v in quat[i].tolist()]
    manifest["settled"] = {"seconds": args.seconds, "still_moving": restless}
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"[settle] {int((moved > 0.03).sum())} objects moved > 3 cm; still moving: {restless or 'none'}", flush=True)
    print(f"[settle] wrote settled poses to {manifest_path}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
