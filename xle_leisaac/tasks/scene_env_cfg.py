"""XLeRobot in a full room, built from a scene manifest. XLeRobot-ReplicaCAD-v0 is the first one.

A scene is three layers, listed in a manifest.json (tools/build_replicacad_usd.py writes one):

  shell         walls, floor, fixed furniture, ceiling lights: one static USD
  objects       movable things (bowls, books, ...): one rigid object collection, reset every episode
  furniture     articulations with drawers and doors (kitchen counter, fridge, ...), fixed base,
                joints damped so a drawer stays where it is left

plus the robot's start pose and named anchors (counter tops, drawer insides) for tasks.
Swapping the apartment for another scene, such as a scan of the lab, means writing another
manifest; robot, cameras, actions, observations and recording stay as they are.

Everything except the scene comes from the sandbox (sandbox_env_cfg.py): the same 17-D
action, the same observations, the same LeIsaac recording hooks. No task, no reward yet.
"""

import json
import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import RenderCfg
from isaaclab.utils import configclass

from xle_leisaac.robots.xlerobot import HEAD_CAMERA_CFG, LEFT_WRIST_CAMERA_CFG, RIGHT_WRIST_CAMERA_CFG, XLEROBOT_CFG

from . import mdp
from .sandbox_env_cfg import XLeRobotSandboxEnvCfg

# Passive furniture joints: no spring, only damping, so drawers and doors stay where they are
# pushed. Units: N*s/m for drawers, N*m*s/rad for doors.
FURNITURE_DAMPING = 5.0


def load_manifest(path: str) -> dict:
    """Read a scene manifest; relative paths in it become absolute."""
    if not os.path.isabs(path):
        path = os.path.join(os.environ["XLE_ASSETS"], path)
    with open(path) as f:
        manifest = json.load(f)
    folder = os.path.dirname(path)
    absolute = lambda p: os.path.normpath(os.path.join(folder, p))  # noqa: E731
    manifest["shell"] = absolute(manifest["shell"])
    for entry in manifest["objects"] + manifest["articulations"]:
        entry["usd"] = absolute(entry["usd"])
    return manifest


def scene_entities(manifest: dict) -> dict:
    """Isaac Lab asset configs for the shell, the movable objects and the furniture."""
    entities = {
        "shell": AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Shell", spawn=sim_utils.UsdFileCfg(usd_path=manifest["shell"]))
    }
    if manifest["objects"]:
        entities["objects"] = RigidObjectCollectionCfg(
            rigid_objects={
                o["name"]: RigidObjectCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/obj_{o['name']}",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=o["usd"],
                        # Things that start slightly inside each other (a knife in its block, a
                        # picture against the wall) drift apart instead of being shot across the room.
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=0.3),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(o["pos"]), rot=tuple(o["rot"])),
                )
                for o in manifest["objects"]
            }
        )
    for a in manifest["articulations"]:
        entities[a["name"]] = ArticulationCfg(
            prim_path=f"{{ENV_REGEX_NS}}/furn_{a['name']}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=a["usd"],
                # No gravity on furniture: ReplicaCAD's hinges are a fraction of a degree off
                # vertical, which is enough for gravity to swing undamped doors open.
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    fix_root_link=a.get("fixed_base", True), enabled_self_collisions=False
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=tuple(a["pos"]), rot=tuple(a["rot"]), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}
            ),
            actuators={
                "passive": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=FURNITURE_DAMPING)
            },
        )
    return entities


@configclass
class ManifestSceneCfg(InteractiveSceneCfg):
    """Robot, cameras and the sky. The rest is added from the manifest in __post_init__ below."""

    robot = XLEROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    head = HEAD_CAMERA_CFG
    left_wrist = LEFT_WRIST_CAMERA_CFG
    right_wrist = RIGHT_WRIST_CAMERA_CFG
    # Daylight through the windows; the rooms are lit by the shell's ceiling lights.
    sky = AssetBaseCfg(
        prim_path="/World/sky", spawn=sim_utils.DomeLightCfg(intensity=400.0, color=(0.85, 0.9, 1.0))
    )


@configclass
class SceneEventsCfg:
    """On reset: robot and furniture back to their start (drawers shut), objects back in place."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_objects = EventTerm(func=mdp.reset_collections_to_default, mode="reset")


@configclass
class XLeRobotSceneEnvCfg(XLeRobotSandboxEnvCfg):
    """XLeRobot in the scene described by `scene_manifest` (a path relative to $XLE_ASSETS)."""

    scene_manifest: str = "replicacad/scenes/apt_0/manifest.json"
    # Apartments are ~13 m across; keep parallel copies apart.
    scene: ManifestSceneCfg = ManifestSceneCfg(num_envs=1, env_spacing=20.0)
    events: SceneEventsCfg = SceneEventsCfg()
    task_description: str = "XLeRobot in an apartment: drive, look around, open drawers, move things."
    # Teleop's third-person camera, in the robot base frame: (pos, rot w-x-y-z, "world" convention).
    # Indoors it has to stay close: the sandbox's 1.6 m behind would sit outside the kitchen wall
    # at the start pose. Here: 0.7 m behind, 1.45 m up, looking 35 degrees down.
    chase_camera_offset: tuple = ((-0.7, 0.0, 1.45), (0.95372, 0.0, 0.30071, 0.0))

    def __post_init__(self):
        super().__post_init__()
        manifest = load_manifest(self.scene_manifest)
        for name, cfg in scene_entities(manifest).items():
            setattr(self.scene, name, cfg)

        spawn = manifest["robot_spawn"]
        yaw = math.radians(spawn.get("yaw_deg", 0.0))
        self.scene.robot = self.scene.robot.replace(
            init_state=self.scene.robot.init_state.replace(
                pos=tuple(spawn["pos"]), rot=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
            )
        )

        # Rendering: shadows, reflections and ambient occlusion on, so the rooms look like rooms.
        self.sim.render = RenderCfg(
            rendering_mode="balanced",
            enable_shadows=True,
            enable_reflections=True,
            enable_ambient_occlusion=True,
            antialiasing_mode="DLAA",
        )
        # Many resting contacts (100+ objects on shelves and tables).
        self.sim.physx.gpu_max_rigid_patch_count = 2**19
        self.viewer.eye = (spawn["pos"][0] - 2.0, spawn["pos"][1] - 2.0, 1.8)
        self.viewer.lookat = (spawn["pos"][0] + 1.0, spawn["pos"][1], 0.6)
