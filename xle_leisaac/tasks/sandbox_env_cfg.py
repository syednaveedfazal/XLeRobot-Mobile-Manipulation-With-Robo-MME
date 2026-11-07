"""XLeRobot-Sandbox-v0: a plain room to drive, look and grab in. No task, no reward.

It is meant as a starting point for your own tasks: copy this file, add a success
condition and your own objects. It follows Isaac Lab's "manager-based" layout, where an
environment is a set of small, named pieces:

  scene         what exists: floor, light, robot, cameras, table, objects
  observations  what the policy sees each step
  actions       how the 17 action values move the robot
  events        what happens on reset (here: shuffle the objects)
  terminations  when an episode ends (here: only after 60 s)
  rewards       empty - this sandbox has no goal

Layout (metres, robot starts at the origin facing +x):
- table: near edge at x = 0.95, top at 0.75 m. The arms are mounted near the back of
  the cart and reach only ~0.15 m past its front, so drive ~0.7 m forward to work at it.
- 3 cubes and a cylinder near the table edge, shuffled on every reset.
- 2 open bins and 2 upside-down cups (with a knob) for hiding and storing things.
- 2 panels on the table and a partition on the floor that block the camera's view.
"""

import math
import os

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from xle_leisaac.robots.xlerobot import (
    BASE_JOINTS,
    HEAD_CAMERA_CFG,
    HEAD_JOINTS,
    LEFT_ARM_JOINTS,
    LEFT_WRIST_CAMERA_CFG,
    RIGHT_ARM_JOINTS,
    RIGHT_WRIST_CAMERA_CFG,
    XLEROBOT_ACTION_KEYS,
    XLEROBOT_CFG,
    sim_to_lerobot_action,
)

from . import mdp

TABLE_SIZE = (0.6, 1.0, 0.75)  # depth (x), width (y), height (z)
TABLE_POS = (0.95 + TABLE_SIZE[0] / 2, 0.0, TABLE_SIZE[2] / 2)
TABLE_TOP = TABLE_SIZE[2]


def _static_box(size, pos, color):
    """A box that collides but never moves (no rigid body)."""
    return AssetBaseCfg(
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


def _object(spawn, pos):
    """A small movable object the robot can pick up."""
    spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg()
    spawn.mass_props = sim_utils.MassPropertiesCfg(mass=0.05)
    spawn.collision_props = sim_utils.CollisionPropertiesCfg()
    return RigidObjectCfg(spawn=spawn, init_state=RigidObjectCfg.InitialStateCfg(pos=pos))


def _cube(color, pos):
    return _object(
        sim_utils.CuboidCfg(size=(0.035, 0.035, 0.035), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color)),
        pos,
    )


def _prop(name, pos):
    return RigidObjectCfg(
        spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(os.environ["XLE_ASSETS"], "props", f"{name}.usd")),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


# --- Scene ---------------------------------------------------------------------------------
@configclass
class SandboxSceneCfg(InteractiveSceneCfg):
    """Everything that exists in the world. {ENV_REGEX_NS} is replaced by each env's own path."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2500.0))

    robot = XLEROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    head = HEAD_CAMERA_CFG
    left_wrist = LEFT_WRIST_CAMERA_CFG
    right_wrist = RIGHT_WRIST_CAMERA_CFG

    table = _static_box(TABLE_SIZE, TABLE_POS, (0.55, 0.4, 0.3)).replace(prim_path="{ENV_REGEX_NS}/table")

    # Objects: start in separate lanes along the table edge so random resets never overlap.
    cube_red = _cube((0.85, 0.1, 0.1), (1.0, -0.2, TABLE_TOP + 0.02)).replace(prim_path="{ENV_REGEX_NS}/cube_red")
    cube_green = _cube((0.1, 0.7, 0.2), (1.0, 0.0, TABLE_TOP + 0.02)).replace(prim_path="{ENV_REGEX_NS}/cube_green")
    cube_blue = _cube((0.1, 0.25, 0.85), (1.0, 0.2, TABLE_TOP + 0.02)).replace(prim_path="{ENV_REGEX_NS}/cube_blue")
    cylinder = _object(
        sim_utils.CylinderCfg(radius=0.015, height=0.05, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.8, 0.1))),
        (1.1, -0.1, TABLE_TOP + 0.03),
    ).replace(prim_path="{ENV_REGEX_NS}/cylinder")

    # Containers (built by tools/build_props_usd.py).
    bin_left = _prop("bin", (1.04, 0.36, TABLE_TOP)).replace(prim_path="{ENV_REGEX_NS}/bin_left")
    bin_right = _prop("bin", (1.04, -0.36, TABLE_TOP)).replace(prim_path="{ENV_REGEX_NS}/bin_right")
    cup_a = _prop("cup", (1.1, 0.1, TABLE_TOP)).replace(prim_path="{ENV_REGEX_NS}/cup_a")
    cup_b = _prop("cup", (1.1, -0.25, TABLE_TOP)).replace(prim_path="{ENV_REGEX_NS}/cup_b")

    # Occluders: two panels on the table and a partition on the floor.
    panel_left = _static_box((0.02, 0.2, 0.2), (1.35, 0.2, TABLE_TOP + 0.1), (0.6, 0.6, 0.65)).replace(
        prim_path="{ENV_REGEX_NS}/panel_left"
    )
    panel_right = _static_box((0.02, 0.2, 0.2), (1.35, -0.2, TABLE_TOP + 0.1), (0.6, 0.6, 0.65)).replace(
        prim_path="{ENV_REGEX_NS}/panel_right"
    )
    partition = _static_box((1.0, 0.03, 1.2), (0.5, 1.2, 0.6), (0.7, 0.7, 0.72)).replace(
        prim_path="{ENV_REGEX_NS}/partition"
    )


# --- Actions -------------------------------------------------------------------------------
@configclass
class ActionsCfg:
    """The 17-D action, in the real robot's key order (see XLEROBOT_ACTION_KEYS).

    Terms are concatenated in the order they are listed here.
    """

    # 14 absolute joint targets in radians: left arm (6), right arm (6), head (2).
    joints = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + HEAD_JOINTS,
        preserve_order=True,
        use_default_offset=False,
        scale=1.0,
    )
    # 3 base speeds in the robot's frame: forward m/s, left m/s, turn rad/s.
    base = mdp.BaseVelocityActionCfg(asset_name="robot", joint_names=BASE_JOINTS)


# --- Observations --------------------------------------------------------------------------
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """What a policy would see: the robot's 17 values and the three camera images."""

        state = ObsTerm(func=mdp.xlerobot_state)
        head = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("head"), "data_type": "rgb", "normalize": False})
        left_wrist = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("left_wrist"), "data_type": "rgb", "normalize": False}
        )
        right_wrist = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("right_wrist"), "data_type": "rgb", "normalize": False}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False  # keep each term separate (images can't be flattened together)

    policy: PolicyCfg = PolicyCfg()


# --- Events --------------------------------------------------------------------------------
def _shuffle(name, dx=(-0.03, 0.03), dy=(-0.06, 0.06)):
    """Move an object by a random offset from its start pose, and spin it, on every reset."""
    return EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": dx, "y": dy, "yaw": (-math.pi, math.pi)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg(name),
        },
    )


@configclass
class EventsCfg:
    """Things that happen at set moments. The random numbers come from env_cfg.seed."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    shuffle_red = _shuffle("cube_red")
    shuffle_green = _shuffle("cube_green")
    shuffle_blue = _shuffle("cube_blue")
    shuffle_cylinder = _shuffle("cylinder", dy=(-0.03, 0.03))


# --- Terminations and rewards --------------------------------------------------------------
@configclass
class TerminationsCfg:
    """No success condition: an episode only ends when time runs out."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class RewardsCfg:
    """Empty on purpose. Add reward terms here if you ever train with RL."""


# --- The environment -----------------------------------------------------------------------
@configclass
class XLeRobotSandboxEnvCfg(ManagerBasedRLEnvCfg):
    scene: SandboxSceneCfg = SandboxSceneCfg(num_envs=1, env_spacing=6.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()

    # Used by LeIsaac's recording and conversion tools.
    robot_name: str = "xlerobot"
    task_description: str = "XLeRobot sandbox: drive, look around, reach and grasp."
    default_feature_joint_names: list[str] = XLEROBOT_ACTION_KEYS
    task_type: str = "scripted"

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 1 / 60  # physics at 60 Hz
        self.decimation = 2  # one action every 2 physics steps -> 30 Hz, like the real cameras
        self.sim.render_interval = self.decimation
        self.episode_length_s = 60.0
        self.viewer.eye = (-1.5, -2.0, 1.8)
        self.viewer.lookat = (1.0, 0.0, 0.7)

    # --- hooks that LeIsaac's tools call -----------------------------------------------------
    def use_teleop_device(self, teleop_device) -> None:
        """LeIsaac swaps the action layout per device; ours is always the 17-D action."""
        self.task_type = teleop_device

    def preprocess_device_action(self, action: dict, teleop_device) -> torch.Tensor:
        """Device output (xlerobot-keyboard) is already the 17-D action."""
        return action["joint_state"].unsqueeze(0) if action["joint_state"].dim() == 1 else action["joint_state"]

    def build_lerobot_frame(self, episode_data, dataset_cfg) -> dict:
        """One recorded step -> one LeRobot frame, in the real robot's units."""
        obs = episode_data._data["obs"]
        action = episode_data._data["actions"][-1].unsqueeze(0)
        frame = {
            "action": sim_to_lerobot_action(action)[0].astype("float32"),
            "observation.state": sim_to_lerobot_action(obs["state"][-1].unsqueeze(0))[0].astype("float32"),
            "task": self.task_description,
        }
        for key in dataset_cfg.features:
            if key.startswith("observation.images."):
                frame[key] = obs[key.split(".")[-1]][-1].cpu().numpy()
        return frame
