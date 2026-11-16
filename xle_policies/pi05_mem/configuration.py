"""Config for pi0.5 with an optional memory module, selected by one flag.

    --policy.type=pi05_mem --policy.memory=none              # plain pi0.5, current frame only
    --policy.type=pi05_mem --policy.memory=framesamp_modul   # RoboMME FrameSamp + Modul

`memory=none` builds exactly LeRobot's pi0.5 (same weights, same forward), so the
no-memory baseline and the memory variants come from the same code and the same
training settings. To add another design, add its name to MEMORY_TYPES and a builder
in memory.py:build_memory.
"""

import math
from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config

# <representation>_<integration>, named as in RoboMME's MME-VLA suite.
MEMORY_TYPES = ("none", "framesamp_modul")


@PreTrainedConfig.register_subclass("pi05_mem")
@dataclass
class PI05MemConfig(PI05Config):
    memory: str = "none"

    # Camera whose past frames form the memory. RoboMME uses one fixed view (front);
    # on XLeRobot that is the head camera. None = observation.images.head if present.
    memory_camera: str | None = None
    # Total memory tokens. FrameSamp: budget / tokens_per_frame frames (512 / 16 = 32),
    # sampled evenly over the episode so far, current frame included.
    memory_budget: int = 512
    # SigLIP gives 16x16 tokens per frame; each frame is pooled to this many (4x4).
    memory_tokens_per_frame: int = 16
    memory_pool: str = "mean"  # "mean" (RoboMME's released config) or "max"
    memory_use_pos_emb: bool = True  # 3D sin-cos (time, row, col) position embedding
    memory_pos_emb_dim: int = 768
    memory_num_heads: int = 4  # cross-attention heads per action-expert layer
    # Pre-computed SigLIP features of every dataset frame (tools/build_memory_cache.py).
    # Needed for training only; at inference frames are encoded online.
    memory_feature_dir: str | None = None

    # RoboMME freezes the vision encoder for all variants, the baseline included. With
    # memory it must stay frozen: cached history features come from these weights.
    freeze_vision_encoder: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.memory not in MEMORY_TYPES:
            raise ValueError(f"memory={self.memory!r}; choose one of {MEMORY_TYPES}")
        if self.memory == "none":
            return
        if not self.freeze_vision_encoder:
            raise ValueError("memory needs freeze_vision_encoder=True (history features are cached)")
        side = math.isqrt(self.memory_tokens_per_frame)
        if side * side != self.memory_tokens_per_frame or 16 % side:
            raise ValueError("memory_tokens_per_frame must be 4, 16, 64 or 256 (a pooled 16x16 grid)")
        if self.memory_budget % self.memory_tokens_per_frame:
            raise ValueError("memory_budget must be a multiple of memory_tokens_per_frame")
        if self.memory_pool not in ("mean", "max"):
            raise ValueError("memory_pool must be 'mean' or 'max'")
        if self.memory_pos_emb_dim % 6:
            raise ValueError("memory_pos_emb_dim must be divisible by 6")

    @property
    def memory_max_frames(self) -> int:
        return self.memory_budget // self.memory_tokens_per_frame
