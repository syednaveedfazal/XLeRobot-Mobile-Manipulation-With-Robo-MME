"""XLeRobot for Isaac Lab: the robot config, its cameras, and action conversions.

XLeRobot is two SO-101 arms and a two-axis head on a mobile base. Like LeKiwi, the
base is not simulated with wheels. Three virtual joints (x, y, rotation) move it
directly, which is far more stable than wheel contact and matches the real
three-omni-wheel base, which can drive in any direction.

The robot USD is built by script from two sources (see tools/build_xlerobot_urdf.py):
- body, cart, neck and head: XLeRobot's own model (simulation/Isaac_sim/xlerobot.zip);
- both arms: the official SO-101 "new_calib" model from TheRobotStudio/SO-ARM100,
  including its wrist-camera mount.

Axes: forward is +x, left is +y.
"""

import math
import os

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import TiledCameraCfg
from leisaac.utils.robot_utils import convert_leisaac_action_to_lerobot, convert_lerobot_action_to_leisaac

XLEROBOT_USD_PATH = os.path.join(os.environ["XLE_ASSETS"], "robots/xlerobot_so101/xlerobot_so101.usd")

# Real hardware (the user's BOM): 17 x FeeTech STS3215-C018 at 12 V.
# 30 kg*cm stall torque = 2.94 N*m; about 0.222 s per 60 degrees = 4.7 rad/s.
STS3215_TORQUE_NM = 2.94
STS3215_SPEED_RAD_S = 4.7

SO101_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
LEFT_ARM_JOINTS = [f"left_arm_{name}" for name in SO101_JOINTS]
RIGHT_ARM_JOINTS = [f"right_arm_{name}" for name in SO101_JOINTS]
HEAD_JOINTS = ["head_pan_joint", "head_tilt_joint"]
BASE_JOINTS = ["root_x_axis_joint", "root_y_axis_joint", "root_z_rotation_joint"]

# Sim joints in the same order as the 17 real action keys below.
XLEROBOT_JOINT_NAMES = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + HEAD_JOINTS + BASE_JOINTS
XLEROBOT_ACTION_KEYS = (
    [f"{name}.pos" for name in LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS]
    + ["head_motor_1.pos", "head_motor_2.pos", "x.vel", "y.vel", "theta.vel"]
)

# SO-101 rest pose (folded), the centre of LeIsaac's SO101_FOLLOWER_REST_POSE_RANGE, in radians.
REST_ARM_POSE = {
    "shoulder_pan": 0.0,
    "shoulder_lift": math.radians(-100.0),
    "elbow_flex": math.radians(90.0),
    "wrist_flex": math.radians(50.0),
    "wrist_roll": 0.0,
    "gripper": math.radians(-10.0),
}

XLEROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=XLEROBOT_USD_PATH,
        # With virtual base joints gravity only pulls on the arms. LeKiwi switches it off
        # for its arm too, because the low SO-101 gains cannot hold the arm up.
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=True,  # the root stays put; the virtual joints do the moving
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            **{f"left_arm_{name}": value for name, value in REST_ARM_POSE.items()},
            **{f"right_arm_{name}": value for name, value in REST_ARM_POSE.items()},
            **{name: 0.0 for name in BASE_JOINTS + HEAD_JOINTS},
        },
    ),
    actuators={
        # Implicit actuators are a spring and damper solved inside PhysX:
        # torque = stiffness * position_error + damping * velocity_error.
        # Gains are LeIsaac's SO-101 values; limits are the real STS3215.
        "arms": ImplicitActuatorCfg(
            joint_names_expr=["(left|right)_arm_(shoulder_pan|shoulder_lift|elbow_flex|wrist_flex|wrist_roll)"],
            effort_limit_sim=STS3215_TORQUE_NM,
            velocity_limit_sim=STS3215_SPEED_RAD_S,
            stiffness=17.8,
            damping=0.60,
        ),
        "grippers": ImplicitActuatorCfg(
            joint_names_expr=["(left|right)_arm_gripper"],
            effort_limit_sim=STS3215_TORQUE_NM,
            velocity_limit_sim=STS3215_SPEED_RAD_S,
            stiffness=17.8,
            damping=0.60,
        ),
        # The head uses the same motor. It gets more gain because the neck links carry
        # the camera; the torque and speed limits stay those of the real STS3215.
        "head": ImplicitActuatorCfg(
            joint_names_expr=HEAD_JOINTS,
            effort_limit_sim=STS3215_TORQUE_NM,
            velocity_limit_sim=STS3215_SPEED_RAD_S,
            stiffness=100.0,
            damping=10.0,
        ),
        # No stiffness and a huge damping turns the joint into a velocity controller:
        # it follows the commanded speed and ignores position.
        "base": ImplicitActuatorCfg(
            joint_names_expr=BASE_JOINTS,
            effort_limit_sim=100000.0,
            velocity_limit_sim=100.0,
            stiffness=0.0,
            damping=100000.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


def _camera(link, name, pos, rot, hfov_deg, width, height, convention="world"):
    # A pinhole camera's horizontal field of view is 2 * atan(aperture / (2 * focal length)).
    focal_length = 10.0
    aperture = 2 * focal_length * math.tan(math.radians(hfov_deg) / 2)
    return TiledCameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{link}/{name}",
        offset=TiledCameraCfg.OffsetCfg(pos=pos, rot=rot, convention=convention),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length,
            horizontal_aperture=aperture,
            vertical_aperture=aperture * height / width,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=width,
        height=height,
        update_period=1 / 30.0,
    )


# Real cameras (the user's BOM), at a reduced resolution with the same aspect ratio:
# - wrists: InnoMaker 720p UVC, about 115-130 degree FOV -> 120 degrees, 16:9.
# - head: Intel RealSense D435 or D455; the model is not confirmed yet. The colour FOV is
#   about 69 x 42 degrees (D435) or 90 x 65 degrees (D455). D435 is used until confirmed.
WRIST_HFOV_DEG, WRIST_SIZE = 120.0, (640, 360)
HEAD_HFOV_DEG, HEAD_SIZE = 69.0, (640, 360)

# Head camera: mount point from XLeRobot's URDF. convention="world" means the camera looks
# along its +x with +z up, which is how that model defines the head camera frame.
HEAD_CAMERA_CFG = _camera(
    "head_tilt_link", "head_camera", (0.055, 0.02, 0.0225), (1.0, 0.0, 0.0, 0.0), HEAD_HFOV_DEG, *HEAD_SIZE
)

# Wrist cameras sit at the official SO-101 wrist_camera_link (from the CAD). In that frame
# the 32 mm UVC module's lens barrel points along +z and ends at z = 26.3 mm, and the jaw
# tip lies 14 degrees off that axis. So the camera sits just in front of the lens (27 mm)
# and looks along +z, rolled so that the jaws appear at the bottom of the image, as on the
# real wrist camera. The rotation is computed from the URDF, not tuned by eye.
WRIST_CAMERA_POS = (0.0, 0.0, 0.027)
WRIST_CAMERA_ROT = (0.38013, 0.59624, -0.38013, 0.59624)
LEFT_WRIST_CAMERA_CFG = _camera(
    "left_arm_wrist_camera_link", "left_wrist_camera", WRIST_CAMERA_POS, WRIST_CAMERA_ROT, WRIST_HFOV_DEG, *WRIST_SIZE
)
RIGHT_WRIST_CAMERA_CFG = _camera(
    "right_arm_wrist_camera_link", "right_wrist_camera", WRIST_CAMERA_POS, WRIST_CAMERA_ROT, WRIST_HFOV_DEG, *WRIST_SIZE
)


# --- Conversions between sim joints and the real robot's 17 action values ---------------
#
# Arms: the official "new_calib" SO-101 model is defined to match LeRobot's calibration,
# so LeIsaac's existing SO-101 conversion applies unchanged (degrees -> normalised range).
# Head: sim limits in degrees mapped to -100..100. head_motor_1 is assumed to be pan and
# head_motor_2 tilt; NOT yet confirmed on the real robot.
_HEAD_LIMITS_DEG = ((-90.0, 90.0), (-43.5, 83.0))


def sim_to_lerobot_action(action):
    """(N, 17) sim action -> (N, 17) real action.

    Sim layout is XLEROBOT_JOINT_NAMES: arm and head positions in radians, then body-frame
    base speeds [forward m/s, left m/s, turn rad/s]. Real layout is XLEROBOT_ACTION_KEYS.
    """
    action = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else np.asarray(action)
    real = np.zeros_like(action)
    real[:, 0:6] = convert_leisaac_action_to_lerobot(action[:, 0:6])
    real[:, 6:12] = convert_leisaac_action_to_lerobot(action[:, 6:12])
    for i, (low, high) in enumerate(_HEAD_LIMITS_DEG):
        real[:, 12 + i] = (np.degrees(action[:, 12 + i]) - low) / (high - low) * 200.0 - 100.0
    real[:, 14:16] = action[:, 14:16]  # m/s in both
    real[:, 16] = np.degrees(action[:, 16])  # rad/s -> deg/s
    return real


def lerobot_to_sim_action(action):
    """(N, 17) real action -> (N, 17) sim action. The inverse of sim_to_lerobot_action."""
    action = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else np.asarray(action)
    sim = np.zeros_like(action)
    sim[:, 0:6] = convert_lerobot_action_to_leisaac(action[:, 0:6])
    sim[:, 6:12] = convert_lerobot_action_to_leisaac(action[:, 6:12])
    for i, (low, high) in enumerate(_HEAD_LIMITS_DEG):
        sim[:, 12 + i] = np.radians((action[:, 12 + i] + 100.0) / 200.0 * (high - low) + low)
    sim[:, 14:16] = action[:, 14:16]
    sim[:, 16] = np.radians(action[:, 16])
    return sim


def body_to_joint_velocity(forward, left, turn, yaw):
    """Body-frame base speeds -> speeds for the x, y, rotation joints.

    The x and y joints sit below the rotation joint, so they move along WORLD axes.
    A forward command therefore has to be rotated by the robot's current yaw.
    """
    x_vel = forward * math.cos(yaw) - left * math.sin(yaw)
    y_vel = forward * math.sin(yaw) + left * math.cos(yaw)
    return x_vel, y_vel, turn
