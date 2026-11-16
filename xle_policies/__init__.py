"""Our policies for LeRobot. Importing this package registers them:

    pi05_mem   pi0.5 with a memory flag (xle_policies/pi05_mem)

LeRobot 0.4.2 looks policies up by name in a fixed list (lerobot.policies.factory.
get_policy_class). We wrap that function so the names below resolve too; the config
side is registered by PreTrainedConfig.register_subclass in each configuration.py.
Use tools/train_policy.py instead of lerobot-train so this import happens first.
"""

from lerobot.policies import factory as _factory

from .pi05_mem import PI05MemConfig, PI05MemPolicy

POLICIES = {"pi05_mem": PI05MemPolicy}

if not getattr(_factory.get_policy_class, "_xle_wrapped", False):
    _lerobot_get_policy_class = _factory.get_policy_class

    def get_policy_class(name: str):
        if name in POLICIES:
            return POLICIES[name]
        return _lerobot_get_policy_class(name)

    get_policy_class._xle_wrapped = True
    _factory.get_policy_class = get_policy_class

__all__ = ["PI05MemConfig", "PI05MemPolicy", "POLICIES"]
