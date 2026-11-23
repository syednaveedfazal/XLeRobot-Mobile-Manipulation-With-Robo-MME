"""XLeRobot support for LeIsaac.

Importing this package registers our environments with gymnasium, so that
gym.make("XLeRobot-Sandbox-v0") / gym.make("XLeRobot-ReplicaCAD-v0") and LeIsaac's tools can find them. Only strings are
registered here, so importing it does not need Isaac Sim to be running yet.
"""

import gymnasium as gym

gym.register(
    id="XLeRobot-Sandbox-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "xle_leisaac.tasks.sandbox_env_cfg:XLeRobotSandboxEnvCfg"},
)

# XLeRobot in a furnished apartment (ReplicaCAD apt_0, converted by setup/06_replicacad.sh).
gym.register(
    id="XLeRobot-ReplicaCAD-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "xle_leisaac.tasks.scene_env_cfg:XLeRobotSceneEnvCfg"},
)

# Place-memory task: take a cup from one of 3 identical stations to a work table, then put it back.
gym.register(
    id="XLeRobot-PutItBack-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": "xle_leisaac.tasks.put_it_back_env_cfg:XLeRobotPutItBackEnvCfg"},
)
