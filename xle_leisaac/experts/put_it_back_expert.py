"""Scripted expert for XLeRobot-PutItBack-v0, to generate training demonstrations.

It knows the right station from the sim (env.put_it_back_station): privileged information
that the policy never sees. The policy only gets cameras and joint states, so to imitate
these demos it has to remember where the cube came from.

It never acts on that knowledge before the head camera could show it: it first looks at the
stations and, if the cube is not yet clearly visible (at least VISIBLE_PX x VISIBLE_PX pixels
in pi0.5's 224 px input, measured by projecting the cube into the head camera), drives slowly
along the row (the same way whichever station holds it) until it is. Only then does it drive
to the cube's station, so every decision in a demo has a visible reason.

Timing carries no information about the station: each episode gets a random cruise speed and
random waits (at the start, after the grasp), drawn independently of it, and the wait at the work
table lasts until a random clock time WAIT_UNTIL drawn per episode (B12). Whatever the station, the
robot leaves the table, and so turns back (the memory decision point), at that drawn time: the far
stations only make the wait shorter. Short random waits alone were not enough: the extra driving to a
far station (~5 s per station) outweighed them, so the time of the turn-back gave the station away.

The plan, one step at a time (each step has a base goal and/or an arm goal, a gripper
state, a head direction and a minimum time):

  look at the stations -> (approach until the cube is clear) -> drive beside its station ->
  reach above the cube -> grasp ->
  lift -> drive back, turn around, drive to the work table -> set the cube down -> let go ->
  wait (the stations are behind the robot now) -> grasp it again -> turn back ->
  drive to the SAME station -> set it down -> let go -> arm back to rest

Grasps use the cube's live position; set-down points are fixed places. Arm targets go
through top-down IK on the robot URDF (arm_ik.py). Outputs the 17-D sim action.
"""

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from isaaclab.utils.math import matrix_from_quat

from xle_leisaac.experts.arm_ik import TopDownArmIK
from xle_leisaac.robots.xlerobot import (
    BASE_JOINTS,
    HEAD_JOINTS,
    LEFT_ARM_JOINTS,
    REST_ARM_POSE,
    RIGHT_ARM_JOINTS,
    SO101_JOINTS,
)
from xle_leisaac.tasks.put_it_back_env_cfg import LAYOUT

# Gripper (rad). OPEN while driving; READY (half open) while reaching for the cube's 2 x 2 x 4 cm
# handle: the moving jaw then hangs ~3.5 cm out from the handle, and the last part of closing is
# sideways. The jaws close like tweezers (a V meeting at the tips) and touch the 2 cm-wide
# handle at ~+0.05 rad. CLOSED is only a little past that: closing
# further wedges the hand up and off the cube (the arm has the same weak gains as the gripper).
# GRIPPER_SQUEEZE caps the command just past where the jaws stopped.
GRIPPER_OPEN, GRIPPER_READY, GRIPPER_CLOSED = 1.0, 0.5, -0.05
GRIPPER_SQUEEZE = 0.05  # rad (> the ~0.02 rad the gripper lags while closing freely)
ARM_SPEED = 1.2  # rad/s per joint
ARM_SPEED_HOLDING = 0.6  # slower while the gripper is closed on the cube
GRIPPER_SPEED = 0.6  # rad/s: close gently instead of snapping shut
BASE_SPEED_RANGE, TURN_SPEED = (0.15, 0.30), 1.2  # cruise m/s (random per episode), rad/s (real robot max 0.3 m/s, 90 deg/s)
BASE_ACCEL, TURN_ACCEL = 0.5, 3.0  # m/s^2, rad/s^2: ramp the speed instead of jumping (would shake the cube loose)
REST = np.array([REST_ARM_POSE[j] for j in SO101_JOINTS])
# The SO-101's tool point (gripper_frame_link) is the FIXED jaw's tip, not the middle of the
# jaws, on its inner face; the moving jaw opens outward (away from the robot). So the fixed
# jaw goes beside the handle on the robot's side: half the 2 cm-wide handle plus 3 mm, along base +y.
FIXED_JAW_OFFSET_Y = 0.013
ARM_OUT = REST.copy()
ARM_OUT[0] = math.pi / 2  # folded arm swung out to the right, above everything on the stations
LOOK_STATIONS, LOOK_RIGHT = (-0.5, 0.45), (-1.25, 0.7)  # head pan, tilt (rad)
# Lifts: the grasp point is 7.7 cm above the 70 cm surfaces and top-down reach ends ~0.85 m.
LIFT, NEAR = 0.06, 0.04  # m above the grasp point: carrying / hovering height, approach height
# Look first: the cube counts as seen when its box is at least VISIBLE_PX in both directions in
# the head image as pi0.5 gets it (640 x 360 scaled to fit 224), for SEEN_STEPS steps in a row.
VISIBLE_PX, SEEN_STEPS = 10.0, 5
POLICY_IMAGE = 224
SEARCH_SPEED, SEARCH_MAX_X = 0.12, 1.1  # m/s forward along the row while looking; stop by here
NOISE_NEAR = (0.08, 0.15)  # m, rad: no base noise this close to a base goal (action_noise.py)
# Clock time (s from the episode start) at which the wait at the work table ends, drawn per episode.
# The robot gets to the table by ~55 s at the latest (D2: turn-back at 37-68 s incl. a 1-9 s wait),
# so the lower end leaves a margin; an episode that arrives later anyway is flagged "late".
WAIT_UNTIL = (60.0, 75.0)
MIN_TABLE_WAIT = 1.0  # s


@dataclass
class Step:
    name: str
    base: tuple | None = None  # (x, y, yaw) goal, env-local
    tool: tuple | str | None = None  # tool point goal: env-local (x, y, z), "rest", "cube+dz:<metres>", or
    # "away:<d>:<dz>" = the previous goal moved d metres away from the cube (fixed-jaw side) and dz up
    gripper: float = GRIPPER_OPEN
    head: tuple = LOOK_RIGHT
    min_time: float = 0.0
    refine: bool = False  # re-aim from the measured tool point (removes the arm's tracking error)
    look: bool = False  # measure how visible the cube is (look at / approach the stations)
    search: bool = False  # drive slowly forward until the cube is seen (skipped if it already is)
    precise: bool = False  # grasping or setting down: no action noise (see action_noise.py)
    q_goal: np.ndarray | None = field(default=None, repr=False)  # filled in when the step starts


class PutItBackExpert:
    def __init__(self, env, seed=0):
        self.env = env
        self.rng = np.random.default_rng(seed)  # timing randomisation (never depends on the station)
        self.robot = env.scene["robot"]
        self.ik = TopDownArmIK("right")
        self.dt = env.step_dt
        ids = lambda names: self.robot.find_joints(names, preserve_order=True)[0]  # noqa: E731
        self.base_ids, self.right_ids = ids(BASE_JOINTS), ids(RIGHT_ARM_JOINTS)
        self.head_cam = env.scene["head"]
        # Where the head camera is, from the physics pose of the link it is mounted on plus its
        # fixed offset. The camera's own stored pose (data.pos_w / quat_w_ros) is read from the USD
        # stage, which is not kept up to date for robot links, so it stays at the start pose.
        cam_cfg = self.head_cam.cfg
        self.head_link = self.robot.find_bodies(cam_cfg.prim_path.split("/")[-2])[0][0]
        self.head_offset_pos = torch.tensor(cam_cfg.offset.pos, device=env.device)
        self.head_offset_rot = matrix_from_quat(torch.tensor(cam_cfg.offset.rot, device=env.device))
        assert cam_cfg.offset.convention == "world", "cube_pixels() assumes the camera looks along +x, z up"
        # world convention (x forward, y left, z up) -> ROS camera axes (x right, y down, z forward)
        self.world_to_ros = torch.tensor([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], device=env.device)

    # --- plan ----------------------------------------------------------------------------------
    def reset(self):
        L = LAYOUT
        self.station = int(self.env.put_it_back_station[0])
        sx = L["STATION_XS"][self.station]
        grasp_h = L["GRASP_HEIGHT"]
        table_x, table_y = L["WORK_TABLE_PLACE"]
        beside_station = (sx + 0.09, 0.0, 0.0)  # right shoulder level with the station
        beside_table = (table_x - 0.09, 0.0, math.pi)  # turned round: the table is on the right
        # Set down so the block's bottom is 2 mm above the surface: dropped from higher, a tall
        # block held slightly tilted can fall over.
        set_down_station = (sx, L["STATION_Y"], L["STATION_TOP"] + grasp_h + 0.002)
        set_down_table = (table_x, table_y, L["WORK_TABLE_TOP"] + grasp_h + 0.002)
        above = lambda p: (p[0], p[1], p[2] + 0.03)  # noqa: E731
        closed, ready = GRIPPER_CLOSED, GRIPPER_READY
        # Random timing per episode, independent of the station (see the module docstring).
        self.base_speed = float(self.rng.uniform(*BASE_SPEED_RANGE))
        self.waits = dict(start=float(self.rng.uniform(1.0, 3.0)), grasp=float(self.rng.uniform(0.0, 2.0)),
                          table=None)  # set when the table wait starts: WAIT_UNTIL minus the time so far
        self.wait_until = float(self.rng.uniform(*WAIT_UNTIL))
        self.late = False
        self.events = {}  # step name -> time (s) it started
        self.plan = [
            Step("look at the stations", tool="rest", head=LOOK_STATIONS, min_time=self.waits["start"], look=True),
            Step("approach until the cube is clear", tool="rest", head=LOOK_STATIONS, look=True, search=True),
            Step("drive to the station", base=beside_station, tool="rest"),
            Step("swing the arm out", tool="out"),
            Step("high above the cube", tool=f"cube+dz:{LIFT}", gripper=ready),
            Step("just above the cube", tool=f"cube+dz:{NEAR}", gripper=ready, min_time=0.3, precise=True),
            Step("down to the cube", tool="cube+dz:0.0", gripper=ready, min_time=0.6, precise=True),
            Step("line up with the cube", tool="cube+dz:0.0", gripper=ready, min_time=0.6, refine=True, precise=True),
            Step("grasp", tool="hold", gripper=closed, min_time=1.0 + self.waits["grasp"], precise=True),
            Step("lift", tool=f"cube+dz:{LIFT}", gripper=closed, precise=True),
            Step("back up", base=(-0.3, 0.0, 0.0), tool="hold", gripper=closed, head=(0.0, 0.3)),
            Step("turn around", base=(-0.3, 0.0, math.pi), tool="hold", gripper=closed, head=(0.0, 0.3)),
            Step("drive to the work table", base=beside_table, tool="hold", gripper=closed),
            # Over the spot first, then straight down: moving the joints straight to the spot can
            # touch the cube to the table early, where it sticks and holds the arm back.
            Step("over the table spot", tool=above(set_down_table), gripper=closed, min_time=0.3, precise=True),
            Step("set the cube down", tool=set_down_table, gripper=closed, min_time=0.3, precise=True),
            Step("let go", tool="hold", gripper=ready, min_time=0.8, precise=True),  # half open: a wide swing can flick it over
            # Back the fixed jaw off the cube before lifting: rising along its side drags it over.
            # Slightly downward, so any drag presses the cube onto the table instead of tipping it.
            Step("back off", tool="away:0.012:-0.002", gripper=ready, min_time=0.2, precise=True),
            Step("hand up and wait", tool=f"away:0.0:{LIFT}"),  # min_time set in _start_step (WAIT_UNTIL)
            Step("just above the cube again", tool=f"cube+dz:{NEAR}", gripper=ready, min_time=0.3, precise=True),
            Step("down to the cube again", tool="cube+dz:0.0", gripper=ready, min_time=0.6, precise=True),
            Step("line up with the cube again", tool="cube+dz:0.0", gripper=ready, min_time=0.6, refine=True, precise=True),
            Step("grasp again", tool="hold", gripper=closed, min_time=1.0, precise=True),
            Step("lift again", tool=f"cube+dz:{LIFT}", gripper=closed, precise=True),
            Step("away from the table", base=(-0.3, 0.0, math.pi), tool="hold", gripper=closed, head=(0.0, 0.3)),
            Step("turn back", base=(-0.3, 0.0, 0.0), tool="hold", gripper=closed, head=(0.0, 0.3)),
            Step("drive to the same station", base=beside_station, tool="hold", gripper=closed),
            Step("over the station", tool=above(set_down_station), gripper=closed, min_time=0.3, precise=True),
            Step("set the cube down on it", tool=set_down_station, gripper=closed, min_time=0.3, precise=True),
            Step("let go of it", tool="hold", gripper=ready, min_time=0.8, precise=True),
            Step("back off from it", tool="away:0.012:-0.002", gripper=ready, min_time=0.2, precise=True),
            Step("hand up", tool=f"away:0.0:{LIFT}", min_time=0.3),
            Step("arm to rest", tool="rest", head=LOOK_STATIONS, min_time=3.0),
        ]
        self.index, self.step_time = 0, 0.0
        state = self.robot.data.joint_pos[0]
        self.right_cmd = state[self.right_ids].cpu().numpy().copy()
        self.head_cmd = np.zeros(2)
        self.base_cmd = np.zeros(3)
        self.roll_goal = 0.0
        self.seen_steps, self.seen = 0, False
        self._base_err = (0.0, 0.0)  # distance (m) and |yaw| error (rad) to the current base goal
        self.seen_px, self.seen_at = (0.0, 0.0), None  # cube size (224 px) when first seen, time (s)
        self.time = 0.0
        self._start_step()

    @property
    def step_name(self):
        return self.plan[self.index].name if self.index < len(self.plan) else "done"

    @property
    def precise(self):
        """True while grasping or setting down (no action noise then)."""
        return self.index < len(self.plan) and self.plan[self.index].precise

    @property
    def driving(self):
        """True while the base is on its way to a goal, still more than NOISE_NEAR from it: base
        noise gets corrected then. Near the goal it is off, so the robot settles quickly."""
        if self.index >= len(self.plan):
            return False
        step = self.plan[self.index]
        if step.search:
            return True
        return step.base is not None and (self._base_err[0] > NOISE_NEAR[0] or self._base_err[1] > NOISE_NEAR[1])

    def episode_info(self):
        """What was hidden or random in this episode, for the dataset's side file."""
        return dict(station=self.station, seen=self.seen, seen_px=[round(v, 1) for v in self.seen_px],
                    seen_at=None if self.seen_at is None else round(self.seen_at, 2),
                    base_speed=round(self.base_speed, 3),
                    waits={k: None if v is None else round(v, 2) for k, v in self.waits.items()},
                    wait_until=round(self.wait_until, 2), late=self.late,
                    turn_back_at=self.events.get("turn back"), events=self.events)

    # --- helpers -------------------------------------------------------------------------------
    def _base_pose(self):
        x, y, yaw = (float(v) for v in self.robot.data.joint_pos[0, self.base_ids])
        return x, y, yaw

    def _to_base(self, p):
        x, y, yaw = self._base_pose()
        dx, dy = p[0] - x, p[1] - y
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([c * dx + s * dy, -s * dx + c * dy, p[2]])

    def _cube_pose(self):
        """Grasp point on the cube (env-local, follows the cube's tilt), cube yaw and tilt (rad)."""
        cube = self.env.scene["cube"].data
        pos = (cube.root_pos_w[0] - self.env.scene.env_origins[0]).cpu().numpy()
        w, x, y, z = (float(v) for v in cube.root_quat_w[0])
        up = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])  # cube's z axis
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return pos + LAYOUT["GRASP_HEIGHT"] * up, yaw, math.acos(max(-1.0, min(1.0, up[2])))

    def _cube_grasp_point(self, dz):
        cube = self._cube_pose()[0]
        return (cube[0], cube[1], cube[2] + dz)

    def _cube_yaw_in_base(self):
        """The cube's yaw relative to the robot, folded into +-45 deg (the cube is square)."""
        rel = self._cube_pose()[1] - self._base_pose()[2]
        return (rel + math.pi / 4) % (math.pi / 2) - math.pi / 4

    def _start_step(self):
        if self.index >= len(self.plan):
            self.events.setdefault("done", round(self.time, 2))
            return
        step = self.plan[self.index]
        self.events[step.name] = round(self.time, 2)
        if step.name == "hand up and wait":  # leave the table at the drawn clock time, whatever the station
            # ~0.5 s for the hand to rise is part of the step; min_time counts from now.
            step.min_time = max(MIN_TABLE_WAIT, self.wait_until - self.time)
            self.waits["table"] = step.min_time
            self.late = self.wait_until - self.time < MIN_TABLE_WAIT
            if self.late:
                print(f"[expert] warning: at the table only at {self.time:.1f} s, after the drawn "
                      f"{self.wait_until:.1f} s (late episode: its turn-back time depends on the drive)", flush=True)
        arm = self.right_cmd[:4]
        # Wrist roll: square to the cube while reaching for it, kept while holding, 0 when folded.
        if step.tool in ("rest", "out"):
            self.roll_goal = 0.0
        elif isinstance(step.tool, str) and step.tool.startswith("cube+dz:") and step.gripper != GRIPPER_CLOSED:
            self.roll_goal = self._cube_yaw_in_base()  # positive roll turns the jaws anticlockwise
        if step.tool == "rest":
            step.q_goal = REST[:4].copy()
        elif step.tool == "out":
            step.q_goal = ARM_OUT[:4].copy()
        elif step.tool == "hold" or step.tool is None:
            step.q_goal = arm.copy()
        elif isinstance(step.tool, str) and step.tool.startswith("away:"):
            d, dz = (float(v) for v in step.tool.split(":")[1:])
            away = np.array([-math.sin(self.roll_goal), math.cos(self.roll_goal), 0.0])  # cube -> fixed jaw
            # From where the hand really is (it can lag its goal), so backing off never pushes the cube.
            start = self._tool_in_base() if d else self._last_ik_target
            local = start + d * away + np.array([0.0, 0.0, dz])
            self._last_ik_target = local
            step.q_goal, err = self.ik.solve(local, arm)
            if err > 0.005:
                print(f"[expert] warning: '{step.name}' is {err * 100:.1f} cm out of reach", flush=True)
        else:
            if isinstance(step.tool, str) and step.tool.startswith("cube+dz:"):
                target = self._cube_grasp_point(float(step.tool.split(":")[1]))
            else:
                target = step.tool
            if step.base is not None:  # arm goal relative to where the base will end up
                bx, by, byaw = step.base
                dx, dy = target[0] - bx, target[1] - by
                c, s = math.cos(byaw), math.sin(byaw)
                local = np.array([c * dx + s * dy, -s * dx + c * dy, target[2]])
            else:
                local = self._to_base(target)
            # fixed jaw beside the cube, not on it: along the cube face's normal (the jaws' roll)
            local[:2] += FIXED_JAW_OFFSET_Y * np.array([-math.sin(self.roll_goal), math.cos(self.roll_goal)])
            if step.refine:  # aim off by however far the hand really is from where it should be
                local = self._last_ik_target + (local - self._tool_in_base())
            self._last_ik_target = local
            step.q_goal, err = self.ik.solve(local, arm)
            if err > 0.005:
                print(f"[expert] warning: '{step.name}' is {err * 100:.1f} cm out of reach", flush=True)
        if step.name.startswith("lift"):
            self._cube_z_before_lift = self._cube_grasp_point(0.0)[2]
        self.step_time = 0.0

    def cube_pixels(self, return_corners=False):
        """Width and height (px) of the cube in the head image as pi0.5 gets it (scaled to fit
        224 x 224), from its corners projected through the head camera; (0, 0) if not in view."""
        cam = self.head_cam.data
        cube = self.env.scene["cube"].data
        link_rot = matrix_from_quat(self.robot.data.body_quat_w[0, self.head_link])
        cam_pos = self.robot.data.body_pos_w[0, self.head_link] + link_rot @ self.head_offset_pos
        cam_rot = link_rot @ self.head_offset_rot @ self.world_to_ros  # columns: camera x, y, z in world
        h = LAYOUT["BOX_SIZE"] / 2
        corners = [(sx * h, sy * h, z) for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, LAYOUT["BOX_SIZE"])]
        corners.append((0.0, 0.0, LAYOUT["BOX_SIZE"] + LAYOUT["HANDLE_SIZE"][2]))  # handle top
        local = torch.tensor(corners, device=self.env.device)
        world = cube.root_pos_w[0] + local @ matrix_from_quat(cube.root_quat_w[0]).T
        in_cam = (world - cam_pos) @ cam_rot  # ROS: x right, y down, z forward
        if bool((in_cam[:, 2] < 0.05).any()):
            return 0.0, 0.0
        k = cam.intrinsic_matrices[0]
        u = k[0, 0] * in_cam[:, 0] / in_cam[:, 2] + k[0, 2]
        v = k[1, 1] * in_cam[:, 1] / in_cam[:, 2] + k[1, 2]
        height_px, width_px = self.head_cam.image_shape
        u0, u1 = float(u.min().clamp(0, width_px)), float(u.max().clamp(0, width_px))
        v0, v1 = float(v.min().clamp(0, height_px)), float(v.max().clamp(0, height_px))
        scale = POLICY_IMAGE / max(width_px, height_px)
        if return_corners:  # for debugging: the projected corners in the full-size head image
            return (u1 - u0) * scale, (v1 - v0) * scale, u.cpu().numpy(), v.cpu().numpy()
        return (u1 - u0) * scale, (v1 - v0) * scale

    def _tool_in_base(self):
        body = self.robot.find_bodies("right_arm_gripper_frame_link")[0][0]
        return self._to_base((self.robot.data.body_pos_w[0, body] - self.env.scene.env_origins[0]).cpu().numpy())

    def _report_grasp(self, step):
        bodies = self.robot.find_bodies(["right_arm_gripper_frame_link", "right_arm_moving_jaw_so101_v1_link"], preserve_order=True)[0]
        origin = self.env.scene.env_origins[0]
        fixed, moving = (self._to_base((self.robot.data.body_pos_w[0, b] - origin).cpu().numpy()) for b in bodies)
        cube = self._to_base(self._cube_grasp_point(0.0))
        grip = float(self.robot.data.joint_pos[0, self.right_ids[5]])
        _, _, tilt = self._cube_pose()
        print(f"[expert] {step.name}: base frame (m) cube {np.round(cube, 3)}  fixed-jaw tip {np.round(fixed, 3)}  "
              f"moving-jaw link {np.round(moving, 3)}  gripper {grip:+.2f} rad (cmd {self.right_cmd[5]:+.2f})  "
              f"cube yaw {math.degrees(self._cube_yaw_in_base()):+.1f} deg, wrist roll {math.degrees(self.right_cmd[4]):+.1f} "
              f"deg, cube tilt {math.degrees(tilt):.1f} deg", flush=True)

    # --- one control step ------------------------------------------------------------------------
    def act(self):
        """The 17-D sim action for this step (left arm, right arm, head, base speeds)."""
        base_goal = np.zeros(3)
        if self.index < len(self.plan):
            step = self.plan[self.index]
            self.step_time += self.dt

            # Arm: move the commanded joints toward the goal at ARM_SPEED.
            speed = ARM_SPEED_HOLDING if step.gripper == GRIPPER_CLOSED else ARM_SPEED
            self.right_cmd[:4] += np.clip(step.q_goal - self.right_cmd[:4], -speed * self.dt, speed * self.dt)
            self.right_cmd[4] += np.clip(self.roll_goal - self.right_cmd[4], -speed * self.dt, speed * self.dt)
            self.right_cmd[5] += np.clip(step.gripper - self.right_cmd[5], -GRIPPER_SPEED * self.dt, GRIPPER_SPEED * self.dt)
            gripper_done = abs(step.gripper - self.right_cmd[5]) < 1e-3
            if step.gripper == GRIPPER_CLOSED:  # squeeze gently: never far past where the jaws stopped
                measured = float(self.robot.data.joint_pos[0, self.right_ids[5]])
                if self.right_cmd[5] <= measured - GRIPPER_SQUEEZE:
                    self.right_cmd[5] = measured - GRIPPER_SQUEEZE
                    gripper_done = True  # the jaws are on the cube
            head_goal = np.array(step.head)
            self.head_cmd += np.clip(head_goal - self.head_cmd, -1.5 * self.dt, 1.5 * self.dt)
            self.time += self.dt

            # Look first: is the cube clearly visible in the head image yet?
            search_done = True
            if step.look and not self.seen:
                w, h = self.cube_pixels()
                self.seen_steps = self.seen_steps + 1 if min(w, h) >= VISIBLE_PX else 0
                if self.seen_steps >= SEEN_STEPS:
                    self.seen, self.seen_px, self.seen_at = True, (w, h), self.time
                    print(f"[expert] cube clearly visible ({w:.0f} x {h:.0f} px at {POLICY_IMAGE} px) "
                          f"after {self.time:.1f} s, during '{step.name}'", flush=True)
            if step.search and not self.seen:  # drive slowly along the row, the same for every station
                if self._base_pose()[0] < SEARCH_MAX_X:
                    base_goal = np.array([SEARCH_SPEED, 0.0, 0.0])
                    search_done = False
                else:
                    print(f"[expert] warning: cube still not clearly visible at x = {SEARCH_MAX_X} m", flush=True)

            arm_done = (np.abs(step.q_goal - self.right_cmd[:4]).max() < 1e-3 and gripper_done
                        and abs(self.roll_goal - self.right_cmd[4]) < 1e-3)
            base_done = True
            if step.base is not None:
                x, y, yaw = self._base_pose()
                ex, ey = step.base[0] - x, step.base[1] - y
                eyaw = math.atan2(math.sin(step.base[2] - yaw), math.cos(step.base[2] - yaw))
                vx, vy = np.clip(2.0 * ex, -self.base_speed, self.base_speed), np.clip(2.0 * ey, -self.base_speed, self.base_speed)
                c, s = math.cos(yaw), math.sin(yaw)
                base_goal = np.array([c * vx + s * vy, -s * vx + c * vy, np.clip(3.0 * eyaw, -TURN_SPEED, TURN_SPEED)])
                self._base_err = (math.hypot(ex, ey), abs(eyaw))
                base_done = math.hypot(ex, ey) < 0.005 and abs(eyaw) < 0.01
                if base_done:
                    base_goal[:] = 0.0
            if arm_done and base_done and search_done and self.step_time >= step.min_time:
                if step.name in ("line up with the cube", "grasp", "let go", "grasp again"):  # where are the jaws vs the cube, really?
                    self._report_grasp(step)
                if step.name.startswith("lift"):  # report whether the cube really came up
                    rise = self._cube_grasp_point(0.0)[2] - self._cube_z_before_lift
                    print(f"[expert] {step.name}: cube rose {rise * 100:.1f} cm", flush=True)
                self.index += 1
                self._start_step()

        limit = np.array([BASE_ACCEL, BASE_ACCEL, TURN_ACCEL]) * self.dt
        self.base_cmd += np.clip(base_goal - self.base_cmd, -limit, limit)
        if not base_goal.any() and np.abs(self.base_cmd).max() < limit.max():
            self.base_cmd[:] = 0.0
        action = np.concatenate([REST, self.right_cmd, self.head_cmd, self.base_cmd]).astype(np.float32)
        return torch.as_tensor(action, device=self.env.device).unsqueeze(0)
