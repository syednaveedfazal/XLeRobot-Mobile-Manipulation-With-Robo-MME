"""XLeRobot-PutItBack-v0: a place-memory task.

Take a small block (2 x 2 x 4 cm, like a 2x2x4 Rubik's cuboid; the code calls it the cube) from one of K identical stations to a work table,
then return it to the station it came from. While the robot is at the work table the stations are behind it, so on the
way back nothing in view says which station was the right one: the policy has to remember.

Layout (metres, env-local frame, robot starts at the origin facing +x):

    stations   K = 3 identical pedestals in a row on the robot's right, at x = 0.7, 1.2, 1.7,
               y = -0.35, top at 0.70 m (the lab's table height). The arms reach ~0.2 m past the
               cart's side, so the robot drives along the row and grasps sideways with its right arm.
    cube       on one station, chosen at random on every reset (seeded), shifted up to 2 cm and
               turned up to 9 degrees. The robot's start is jittered too.
    work table a four-legged table, 60 x 40 cm top at 0.70 m, behind the start at x = -0.9. The
               robot turns around, so the table is on its right again and the stations are
               behind it, out of the cameras' view.

Success: after the cube has been set down on the work table, it rests again on the station it
started on and the gripper has let go. Failure: the cube is set down on a wrong station, or
dropped. Otherwise the episode times out after 90 s.

Everything else (17-D action, observations, LeIsaac recording hooks) is the sandbox's.
"""

import os

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from xle_leisaac.robots.xlerobot import BASE_JOINTS, HEAD_CAMERA_CFG, LEFT_WRIST_CAMERA_CFG, RIGHT_WRIST_CAMERA_CFG, XLEROBOT_CFG

from . import mdp
from .sandbox_env_cfg import XLeRobotSandboxEnvCfg

# --- Layout --------------------------------------------------------------------------------
K = 3
STATION_XS = (0.7, 1.2, 1.7)
STATION_Y = -0.35
STATION_TOP = 0.70
STATION_SIZE = (0.14, 0.14, STATION_TOP)
WORK_TABLE_PLACE = (-0.9, 0.35)  # where the cube is set down on the work table
WORK_TABLE_TOP = 0.70  # tools/build_props_usd.py work_table(): 60 x 40 cm top, four legs
WORK_TABLE_CENTRE = (WORK_TABLE_PLACE[0], WORK_TABLE_PLACE[1] + 0.12)  # near edge 8 cm inside the place spot, 4 cm off the cart
BLOCK_SIZE = (0.02, 0.02, 0.04)  # tools/build_props_usd.py BLOCK: upright 2 x 2 x 4 cm
GRASP_HEIGHT = 0.017  # tool point above the block's bottom: pinched above its low centre of mass, jaw tips 1.1 cm above the surface
STATION_COLOR = (0.62, 0.62, 0.66)
# Reset randomisation, so demonstrations differ beyond the station choice (uniform, +-):
CUBE_JITTER = (0.02, 0.015, 0.15)  # cube on its station: x, y (m), yaw (rad)
BASE_JITTER = (0.05, 0.03, 0.08)  # robot start: x, y (m), yaw (rad)


def _static_box(size, pos, color):
    return AssetBaseCfg(
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
    )


# --- Scene ---------------------------------------------------------------------------------
@configclass
class PutItBackSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2500.0))

    robot = XLEROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    head = HEAD_CAMERA_CFG
    left_wrist = LEFT_WRIST_CAMERA_CFG
    right_wrist = RIGHT_WRIST_CAMERA_CFG

    # K identical stations (same size, same colour) in a row on the robot's right.
    station_0 = _static_box(STATION_SIZE, (STATION_XS[0], STATION_Y, STATION_TOP / 2), STATION_COLOR).replace(
        prim_path="{ENV_REGEX_NS}/station_0"
    )
    station_1 = _static_box(STATION_SIZE, (STATION_XS[1], STATION_Y, STATION_TOP / 2), STATION_COLOR).replace(
        prim_path="{ENV_REGEX_NS}/station_1"
    )
    station_2 = _static_box(STATION_SIZE, (STATION_XS[2], STATION_Y, STATION_TOP / 2), STATION_COLOR).replace(
        prim_path="{ENV_REGEX_NS}/station_2"
    )
    work_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/work_table",
        spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(os.environ["XLE_ASSETS"], "props", "work_table.usd")),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(WORK_TABLE_CENTRE[0], WORK_TABLE_CENTRE[1], 0.0)),
    )

    # The block (tools/build_props_usd.py: 2 x 2 x 4 cm, 60 g, low centre of mass). Its start
    # pose is set by the reset event below.
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(os.environ["XLE_ASSETS"], "props", "cube.usd"),
            # A light object squeezed by a fast jaw gets pushed out of the overlap at any speed by
            # default and flies off. Cap that speed and give its contacts more solver iterations.
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=0.3,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(STATION_XS[0], STATION_Y, STATION_TOP)),
    )

    # Fixed overview camera for videos (not part of the policy's observations).
    overview = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/overview",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.4, -2.9, 2.3)),  # aimed with set_world_poses_from_view by the tools
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=10.0, clipping_range=(0.05, 30.0)),
        width=640,
        height=360,
    )


# --- Events --------------------------------------------------------------------------------
def place_cube_at_random_station(env, env_ids: torch.Tensor):
    """Choose a station per env (from the seeded torch RNG) and put the cube on it, shifted
    and turned a little (CUBE_JITTER).

    The chosen index is kept on the env as `env.put_it_back_station` (one int per env). It
    is task state for the success check and for the scripted expert, never an observation.
    """
    if not hasattr(env, "put_it_back_station"):
        env.put_it_back_station = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    choice = torch.randint(0, K, (len(env_ids),), device=env.device)
    env.put_it_back_station[env_ids] = choice
    cube = env.scene["cube"]
    xs = torch.tensor(STATION_XS, device=env.device)[choice]
    n = len(env_ids)
    jitter = (torch.rand(n, 3, device=env.device) * 2 - 1) * torch.tensor(CUBE_JITTER, device=env.device)
    pose = torch.zeros(n, 7, device=env.device)
    pose[:, 0] = xs + jitter[:, 0]
    pose[:, 1] = STATION_Y + jitter[:, 1]
    pose[:, 2] = STATION_TOP + 0.001
    pose[:, 3] = torch.cos(jitter[:, 2] / 2)  # quaternion (w, x, y, z), turned about z
    pose[:, 6] = torch.sin(jitter[:, 2] / 2)
    pose[:, :3] += env.scene.env_origins[env_ids]
    cube.write_root_pose_to_sim(pose, env_ids=env_ids)
    cube.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids)


def jitter_robot_start(env, env_ids: torch.Tensor):
    """Start the robot a little off the origin (BASE_JITTER) on its virtual base joints."""
    robot = env.scene["robot"]
    ids = robot.find_joints(BASE_JOINTS, preserve_order=True)[0]
    n = len(env_ids)
    pos = (torch.rand(n, 3, device=env.device) * 2 - 1) * torch.tensor(BASE_JITTER, device=env.device)
    robot.write_joint_state_to_sim(pos, torch.zeros_like(pos), joint_ids=ids, env_ids=env_ids)
    robot.set_joint_position_target(pos, joint_ids=ids, env_ids=env_ids)


@configclass
class PutItBackEventsCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    place_cube = EventTerm(func=place_cube_at_random_station, mode="reset")
    jitter_start = EventTerm(func=jitter_robot_start, mode="reset")


# --- Task progress, success and failure ----------------------------------------------------
def cube_state(env):
    """Cube position (env-local), whether it rests, and whether the right gripper has let go."""
    cube = env.scene["cube"]
    pos = cube.data.root_pos_w - env.scene.env_origins
    resting = torch.linalg.norm(cube.data.root_lin_vel_w, dim=-1) < 0.05
    robot = env.scene["robot"]
    tool = robot.data.body_pos_w[:, robot.find_bodies("right_arm_gripper_frame_link")[0][0]] - env.scene.env_origins
    grasp_point = pos + torch.tensor([0.0, 0.0, GRASP_HEIGHT], device=env.device)
    released = torch.linalg.norm(tool - grasp_point, dim=-1) > 0.06
    return pos, resting, released


def _on_surface(pos, x, y, top, half=0.05):
    return (torch.abs(pos[:, 0] - x) < half) & (torch.abs(pos[:, 1] - y) < half) & (torch.abs(pos[:, 2] - top) < 0.02)


class PutItBackProgress(ManagerTermBase):
    """Tracks whether the cube has been delivered to the work table, then judges the return.

    Used for three termination terms (mode = "success", "wrong_station", "dropped") that all
    share the episode's delivered flag, kept on the env as `env.put_it_back_delivered`.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if not hasattr(env, "put_it_back_delivered"):
            env.put_it_back_delivered = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        self._env.put_it_back_delivered[slice(None) if env_ids is None else env_ids] = False

    def __call__(self, env, mode: str):
        pos, resting, released = cube_state(env)
        flags = env.put_it_back_delivered
        on_table = resting & released & _on_surface(pos, WORK_TABLE_PLACE[0], WORK_TABLE_PLACE[1], WORK_TABLE_TOP, half=0.12)
        flags |= on_table
        if mode == "dropped":
            return pos[:, 2] < 0.3
        station = getattr(env, "put_it_back_station", torch.zeros(env.num_envs, dtype=torch.long, device=env.device))
        on_station = torch.stack([_on_surface(pos, x, STATION_Y, STATION_TOP) for x in STATION_XS], dim=-1)
        on_right = on_station.gather(1, station[:, None])[:, 0]
        on_any = on_station.any(dim=-1)
        settled = flags & resting & released
        if mode == "success":
            return settled & on_right
        return settled & on_any & ~on_right  # wrong_station


@configclass
class PutItBackTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=PutItBackProgress, params={"mode": "success"})
    wrong_station = DoneTerm(func=PutItBackProgress, params={"mode": "wrong_station"})
    dropped = DoneTerm(func=PutItBackProgress, params={"mode": "dropped"})


# --- The environment -----------------------------------------------------------------------
@configclass
class XLeRobotPutItBackEnvCfg(XLeRobotSandboxEnvCfg):
    scene: PutItBackSceneCfg = PutItBackSceneCfg(num_envs=1, env_spacing=8.0)
    events: PutItBackEventsCfg = PutItBackEventsCfg()
    terminations: PutItBackTerminationsCfg = PutItBackTerminationsCfg()
    task_description: str = "Take the cube to the work table, then put it back on the station it came from."

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 90.0
        self.viewer.eye = (0.4, -2.9, 2.3)
        self.viewer.lookat = (0.4, 0.0, 0.6)
        # The overview camera is for our videos only, never recorded. LeIsaac's converter turns
        # every camera in the scene into a dataset feature, so tools/convert_to_lerobot.py sets
        # XLE_VIDEO_CAMERA=0 to build the scene without it.
        if os.environ.get("XLE_VIDEO_CAMERA", "1") == "0":
            self.scene.overview = None


# Exposed for the scripted expert.
LAYOUT = dict(
    K=K, STATION_XS=STATION_XS, STATION_Y=STATION_Y, STATION_TOP=STATION_TOP, WORK_TABLE_PLACE=WORK_TABLE_PLACE,
    WORK_TABLE_TOP=WORK_TABLE_TOP, GRASP_HEIGHT=GRASP_HEIGHT,
)  # fmt: skip
