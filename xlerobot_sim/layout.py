"""Action / state layout of the real XLeRobot, and how it maps onto the ManiSkill model.

The real robot (XLeRobot software, `robots/xlerobot/xlerobot.py`, `_state_ft`) uses 17 keys for
both its action and its state, in this order:

    [0:6]   left arm   shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
    [6:12]  right arm  (same joints)
    [12:14] head       head_motor_1, head_motor_2
    [14:17] base       x.vel, y.vel, theta.vel   (robot frame, x forward, y left)

The control mode ``pd_joint_pos_real_layout`` (see `xlerobot_sim.agent`) uses the same order and
size. Units are NOT the real robot's yet:

    joints    absolute joint targets in radians, in the ManiSkill URDF's joint convention
              (real robot: -100..100, gripper 0..100, after its own calibration)
    x/y.vel   m/s  (same as real)
    theta.vel rad/s (real robot: deg/s)

Converting sim joint angles to real calibrated values still needs a per-joint offset/sign table:
the ManiSkill arm is the older SO-100-style URDF whose zero points differ from the SO-101.
"""

from __future__ import annotations

import math

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")

# ManiSkill URDF joint names. "arm 1" (no suffix) is the RIGHT arm (mounted at y = -0.133 m),
# "arm 2" (suffix _2) is the LEFT arm.
SIM_RIGHT_ARM = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
SIM_LEFT_ARM = tuple(f"{n}_2" for n in SIM_RIGHT_ARM)
# Assumption, not verified on hardware: head_motor_1 = pan, head_motor_2 = tilt.
SIM_HEAD = ("head_pan_joint", "head_tilt_joint")
SIM_BASE = ("root_x_axis_joint", "root_y_axis_joint", "root_z_rotation_joint")

# position-controlled joints, in layout order (indices 0..13)
SIM_POSITION_JOINTS = SIM_LEFT_ARM + SIM_RIGHT_ARM + SIM_HEAD

ACTION_KEYS = (
    [f"left_arm_{j}.pos" for j in ARM_JOINTS]
    + [f"right_arm_{j}.pos" for j in ARM_JOINTS]
    + ["head_motor_1.pos", "head_motor_2.pos"]
    + ["x.vel", "y.vel", "theta.vel"]
)
ACTION_DIM = len(ACTION_KEYS)  # 17

LEFT_ARM = slice(0, 6)
RIGHT_ARM = slice(6, 12)
HEAD = slice(12, 14)
BASE = slice(14, 17)
LEFT_GRIPPER, RIGHT_GRIPPER = 5, 11
HEAD_PAN, HEAD_TILT = 12, 13
BASE_X, BASE_Y, BASE_YAW = 14, 15, 16

# Real base speed presets (xlerobot.py): slow 0.1 m/s / 30 deg/s, medium 0.2 / 60, fast 0.3 / 90.
BASE_MAX_XY = 0.3                   # m/s
BASE_MAX_YAW = math.radians(90.0)   # rad/s

REAL_LAYOUT_MODE = "pd_joint_pos_real_layout"
