"""XLeRobot for Isaac Lab: the robot config, its cameras, and action conversions.

XLeRobot is two SO-101 arms and a two-axis head on a mobile base. Like LeKiwi, the
base is not simulated with wheels. Three virtual joints (x, y, rotation) move it
directly, which is far more stable than wheel contact and matches an omni-wheel
base that can drive in any direction.

Axes: forward is +x, left is +y. The arm with the `_2` suffix is the LEFT arm.
"""

import math
import os

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import TiledCameraCfg
from leisaac.assets.robots.lerobot import SO101_FOLLOWER_MOTOR_LIMITS, SO101_FOLLOWER_USD_JOINT_LIMLITS

XLEROBOT_USD_PATH = os.path.join(os.environ["XLE_ASSETS"], "robots/xlerobot/xlerobot/xlerobot/xlerobot.usd")

ARM_JOINTS = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
BASE_JOINTS = ["root_x_axis_joint", "root_y_axis_joint", "root_z_rotation_joint"]
HEAD_JOINTS = ["head_pan_joint", "head_tilt_joint"]
LEFT_ARM_JOINTS = [f"{name}_2" for name in ARM_JOINTS]
RIGHT_ARM_JOINTS = list(ARM_JOINTS)

# Sim joints in the same order as the 17 real action keys below.
XLEROBOT_JOINT_NAMES = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + HEAD_JOINTS + BASE_JOINTS

_SO101 = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
XLEROBOT_ACTION_KEYS = (
    [f"left_arm_{name}.pos" for name in _SO101]
    + [f"right_arm_{name}.pos" for name in _SO101]
    + ["head_motor_1.pos", "head_motor_2.pos", "x.vel", "y.vel", "theta.vel"]
)

# Folded rest pose, taken from the XLeRobot ManiSkill agent (radians).
REST_ARM_POSE = {"Rotation": 0.0, "Pitch": 3.14, "Elbow": 3.14, "Wrist_Pitch": 0.0, "Wrist_Roll": 1.57, "Jaw": 0.0}

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
            **REST_ARM_POSE,
            **{f"{name}_2": value for name, value in REST_ARM_POSE.items()},
            **{name: 0.0 for name in BASE_JOINTS + HEAD_JOINTS},
        },
    ),
    actuators={
        # Implicit actuators are a spring and damper solved inside PhysX:
        # torque = stiffness * position_error + damping * velocity_error.
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[name for name in LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS if "Jaw" not in name],
            effort_limit_sim=10.0,
            velocity_limit_sim=10.0,
            stiffness=17.8,
            damping=0.60,
        ),
        "grippers": ImplicitActuatorCfg(
            joint_names_expr=["Jaw", "Jaw_2"],
            effort_limit_sim=10.0,
            velocity_limit_sim=10.0,
            stiffness=17.8,
            damping=0.60,
        ),
        # The head links are much heavier than the arm links, so they need more gain.
        "head": ImplicitActuatorCfg(
            joint_names_expr=HEAD_JOINTS,
            effort_limit_sim=50.0,
            velocity_limit_sim=5.0,
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


def _camera_spawn():
    return sim_utils.PinholeCameraCfg(
        focal_length=36.5,
        focus_distance=400.0,
        horizontal_aperture=36.83,  # about 75 degrees, the same as LeIsaac's cameras
        clipping_range=(0.01, 50.0),
        lock_camera=True,
    )


def _camera(link, name, pos, rot):
    # convention="world" means the camera looks along its +x with +z up, which is how
    # the ManiSkill model defines these frames.
    return TiledCameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{link}/{name}",
        offset=TiledCameraCfg.OffsetCfg(pos=pos, rot=rot, convention="world"),
        data_types=["rgb"],
        spawn=_camera_spawn(),
        width=640,
        height=480,
        update_period=1 / 30.0,
    )


# Mount points come from the URDF in xlerobot.zip. Quaternions are (w, x, y, z).
HEAD_CAMERA_CFG = _camera("head_tilt_link", "head_camera", (0.055, 0.02, 0.0225), (1.0, 0.0, 0.0, 0.0))

# The jaw tips point along the Fixed_Jaw link's -y axis. This rotation makes the wrist
# camera look along -y, tilted 20 degrees down toward the jaws, with the link's +z as
# "up" in the image. (The URDF's own camera frame gave a 90-degree rolled image.)
_WRIST_ROT = (0.69636, 0.12279, 0.12279, -0.69636)
LEFT_WRIST_CAMERA_CFG = _camera("Fixed_Jaw_2", "left_wrist_camera", (0.0, -0.02, 0.05), _WRIST_ROT)
RIGHT_WRIST_CAMERA_CFG = _camera("Fixed_Jaw", "right_wrist_camera", (0.0, -0.02, 0.05), _WRIST_ROT)


# --- Conversions between sim joints and the real robot's 17 action values ---------------
#
# Arms go in two steps: sim radians -> SO-101 degrees (a sign and an offset per joint),
# then LeIsaac's existing SO-101 table -> normalised motor range.
# The signs and offsets below were derived from pose comparisons and are NOT yet
# confirmed on the real robot: real = sign * sim + offset, in degrees.
_ARM_SIGN_OFFSET = {
    "shoulder_pan": (1.0, 0.0),
    "shoulder_lift": (-1.0, 90.0),
    "elbow_flex": (1.0, -90.0),
    "wrist_flex": (1.0, 0.0),
    "wrist_roll": (1.0, -90.0),
}
_GRIPPER_SIM_MAX_DEG = 97.4  # sim 0..97.4 deg maps to real 0..100

# Head: sim limits in degrees mapped to the normalised motor range. head_motor_1 is
# assumed to be pan and head_motor_2 tilt.
_HEAD_LIMITS_DEG = ((-90.0, 90.0), (-43.5, 83.0))


def _arm_sim_to_real(rad):
    """(N, 6) sim radians for one arm -> (N, 6) real normalised values."""
    deg = np.degrees(rad)
    real = np.zeros_like(deg)
    for i, name in enumerate(_SO101[:5]):
        sign, offset = _ARM_SIGN_OFFSET[name]
        low, high = SO101_FOLLOWER_USD_JOINT_LIMLITS[name]
        motor_low, motor_high = SO101_FOLLOWER_MOTOR_LIMITS[name]
        real[:, i] = (sign * deg[:, i] + offset - low) / (high - low) * (motor_high - motor_low) + motor_low
    real[:, 5] = deg[:, 5] / _GRIPPER_SIM_MAX_DEG * 100.0
    return real


def _arm_real_to_sim(real):
    """(N, 6) real normalised values for one arm -> (N, 6) sim radians."""
    deg = np.zeros_like(real)
    for i, name in enumerate(_SO101[:5]):
        sign, offset = _ARM_SIGN_OFFSET[name]
        low, high = SO101_FOLLOWER_USD_JOINT_LIMLITS[name]
        motor_low, motor_high = SO101_FOLLOWER_MOTOR_LIMITS[name]
        so101_deg = (real[:, i] - motor_low) / (motor_high - motor_low) * (high - low) + low
        deg[:, i] = (so101_deg - offset) / sign
    deg[:, 5] = real[:, 5] / 100.0 * _GRIPPER_SIM_MAX_DEG
    return np.radians(deg)


def sim_to_lerobot_action(action):
    """(N, 17) sim action -> (N, 17) real action.

    Sim layout is XLEROBOT_JOINT_NAMES: arm and head positions in radians, then body-frame
    base speeds [forward m/s, left m/s, turn rad/s]. Real layout is XLEROBOT_ACTION_KEYS.
    """
    action = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else np.asarray(action)
    real = np.zeros_like(action)
    real[:, 0:6] = _arm_sim_to_real(action[:, 0:6])
    real[:, 6:12] = _arm_sim_to_real(action[:, 6:12])
    for i, (low, high) in enumerate(_HEAD_LIMITS_DEG):
        real[:, 12 + i] = (np.degrees(action[:, 12 + i]) - low) / (high - low) * 200.0 - 100.0
    real[:, 14:16] = action[:, 14:16]  # m/s in both
    real[:, 16] = np.degrees(action[:, 16])  # rad/s -> deg/s
    return real


def lerobot_to_sim_action(action):
    """(N, 17) real action -> (N, 17) sim action. The inverse of sim_to_lerobot_action."""
    action = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else np.asarray(action)
    sim = np.zeros_like(action)
    sim[:, 0:6] = _arm_real_to_sim(action[:, 0:6])
    sim[:, 6:12] = _arm_real_to_sim(action[:, 6:12])
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
