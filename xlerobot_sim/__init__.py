"""XLeRobot <-> ManiSkill glue.

`import xlerobot_sim` registers, without touching the installed mani_skill package:
  * agent  "xlerobot"               (dual-arm XLeRobot, from third_party/xlerobot_maniskill)
  * scene  "XLeReplicaCAD"          (ReplicaCAD scene builder that knows how to place XLeRobot)
  * env    "XLeRobot-ReplicaCAD-v1" (ReplicaCAD apartment + XLeRobot, no task/reward)

The upstream XLeRobot instructions instead copy files into site-packages/mani_skill and hand-edit
`agents/robots/__init__.py` and the ReplicaCAD scene builder; this module does the same at import time.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import sapien
import torch

from mani_skill.envs.scenes.base_env import SceneManipulationEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.registration import register_scene_builder
from mani_skill.utils.scene_builder.replicacad.scene_builder import ReplicaCADSceneBuilder

REPO_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_DIR = REPO_ROOT / "third_party" / "xlerobot_maniskill"
XLEROBOT_URDF = UPSTREAM_DIR / "assets" / "xlerobot" / "xlerobot.urdf"

ENV_ID = "XLeRobot-ReplicaCAD-v1"


def _load_upstream_agent():
    """Import the vendored agent module (its @register_agent registers uid 'xlerobot')."""
    name = "xlerobot_sim._upstream_xlerobot_agent"
    if name in sys.modules:
        return sys.modules[name]
    path = UPSTREAM_DIR / "agents" / "xlerobot" / "xlerobot.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    # upstream expects the URDF inside mani_skill's package assets; point it at our copy instead
    module.Xlerobot.urdf_path = str(XLEROBOT_URDF)
    return module


Xlerobot = _load_upstream_agent().Xlerobot


@register_scene_builder("XLeReplicaCAD")
class XLeReplicaCADSceneBuilder(ReplicaCADSceneBuilder):
    """ReplicaCAD apartment; places XLeRobot where ManiSkill places Fetch."""

    xlerobot_spawn_pose = sapien.Pose([-1.0, 0.0, 0.02])

    def initialize(self, env_idx: torch.Tensor):
        try:
            super().initialize(env_idx)  # resets all scene objects, then rejects non-Fetch robots
        except NotImplementedError:
            if self.env.robot_uids != "xlerobot":
                raise
            self.env.agent.reset(self.env.agent.keyframes["rest"].qpos)
            self.env.agent.robot.set_pose(self.xlerobot_spawn_pose)


@register_env(ENV_ID, max_episode_steps=100_000)
class XLeRobotSceneEnv(SceneManipulationEnv):
    """ReplicaCAD scene with XLeRobot. Exploration/teleop only: no reward, no success."""

    SUPPORTED_ROBOTS = ["xlerobot"]

    def __init__(self, *args, robot_uids="xlerobot", scene_builder_cls="XLeReplicaCAD", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, scene_builder_cls=scene_builder_cls, **kwargs)

    @property
    def _default_sensor_configs(self):
        return []  # robot cameras (head + 2 wrists) come from the agent itself

    @property
    def _default_human_render_camera_configs(self):
        # Chase camera behind/above the robot, looking where it drives (mounted on top_base_link, z~0.75 m).
        # Upstream used look_at([1, 0, 0.6], [0, 0, 0.3]) -- in front, looking back -- which ends up
        # inside walls as soon as the robot drives toward them.
        pose = sapien_utils.look_at([-1.0, 0, 1.0], [0.5, 0, 0.3])
        return [CameraConfig("render_camera", pose, 640, 480, 1.2, 0.01, 100, mount=self.agent.top_base_link)]


__all__ = ["ENV_ID", "Xlerobot", "XLeReplicaCADSceneBuilder", "XLeRobotSceneEnv"]
