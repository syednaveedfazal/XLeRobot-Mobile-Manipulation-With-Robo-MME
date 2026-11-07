"""build_xlerobot_usd.py - convert the combined XLeRobot URDF into a USD for Isaac Sim.

Input:  $XLE_ASSETS/robots/xlerobot_so101/xlerobot_so101.urdf  (from build_xlerobot_urdf.py)
Output: $XLE_ASSETS/robots/xlerobot_so101/xlerobot_so101.usd

Converter choices:
- merge_fixed_joints=False keeps small fixed links such as wrist_camera_link, which the
  cameras attach to.
- fix_base=False: the robot config fixes the root link instead (fix_root_link=True).
- Joint gains are left at the converter defaults; the actuators in
  xle_leisaac/robots/xlerobot.py set the real ones at run time.

  slurm/in_container.sh /isaac-sim/python.sh tools/build_xlerobot_usd.py
"""

import os

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

FOLDER = os.path.join(os.environ["XLE_ASSETS"], "robots/xlerobot_so101")


def main():
    cfg = UrdfConverterCfg(
        asset_path=os.path.join(FOLDER, "xlerobot_so101.urdf"),
        usd_dir=FOLDER,
        usd_file_name="xlerobot_so101.usd",
        fix_base=False,
        merge_fixed_joints=False,
        make_instanceable=True,
        force_usd_conversion=True,
        self_collision=False,
        collider_type="convex_hull",
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    )
    converter = UrdfConverter(cfg)
    print(f"[usd] wrote {converter.usd_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
