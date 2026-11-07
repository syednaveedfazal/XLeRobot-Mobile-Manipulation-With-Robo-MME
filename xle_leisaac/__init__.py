"""XLeRobot support for LeIsaac.

Importing this package registers our environments with gymnasium, so that
gym.make("XLeRobot-Sandbox-v0") and LeIsaac's tools can find them. Only strings are
registered here, so importing it does not need Isaac Sim to be running yet.
"""

import gymnasium as gym

gym.register(
    id="XLeRobot-Sandbox-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "xle_leisaac.tasks.sandbox_env_cfg:XLeRobotSandboxEnvCfg"},
)
