"""pi0.5 with optional memory: LeRobot's PI05Policy plus a memory module from memory.py.

With config.memory == "none" this is LeRobot's pi0.5 unchanged (apart from the frozen
vision encoder, see PI05MemConfig.freeze_vision_encoder).

Where memory frames come from
  training   batch["index"] (the dataset row, kept by LeRobot's processors) -> frames
             sampled from the cached SigLIP features of that episode up to that row.
  inference  select_action / predict_action_chunk encode the memory camera's frame at
             every call and keep it in an online buffer; reset() clears it. Call
             select_action every control step (at the dataset's fps), as the policy was
             trained with one memory frame per dataset frame.
"""

import dataclasses
import logging

import torch
from torch import Tensor

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy, resize_with_pad_torch

from .configuration import PI05MemConfig
from .memory import MemoryFeatureCache, build_memory, frame_sampling_indices, image_feature_dim, pool_tokens

MEMORY_PREFIX = "model.memory."


class PI05MemPolicy(PI05Policy):
    config_class = PI05MemConfig
    name = "pi05_mem"

    def __init__(self, config: PI05MemConfig):
        super().__init__(config)
        if config.freeze_vision_encoder:
            pg = self.model.paligemma_with_expert.paligemma.model
            for p in [*pg.vision_tower.parameters(), *pg.multi_modal_projector.parameters()]:
                p.requires_grad_(False)

        # Inside self.model so checkpoint keys are model.memory.* (PI05Policy.from_pretrained
        # prefixes every key without "model.").
        self.model.memory = build_memory(config, self.model)
        if self.model.memory is not None:
            self.model.memory.to(next(self.model.parameters()).device)
            if config.memory_camera is None:
                cams = list(config.image_features)
                config.memory_camera = "observation.images.head" if "observation.images.head" in cams else cams[0]
            logging.info(
                "pi05_mem: memory=%s camera=%s frames=%d params=%.1fM",
                config.memory,
                config.memory_camera,
                config.memory_max_frames,
                sum(p.numel() for p in self.model.memory.parameters()) / 1e6,
            )
        self._feature_cache = None
        self._load_report = None
        self.reset()

    @property
    def uses_memory(self) -> bool:
        return getattr(self.model, "memory", None) is not None

    # ------------------------------------------------------------------ frames -> features

    @torch.no_grad()
    def encode_memory_frames(self, images: Tensor) -> Tensor:
        """(B, 3, H, W) images in [0, 1] -> (B, tokens_per_frame, D) float16 pooled SigLIP
        features. Used both to build the training cache and online, so the two match."""
        device = next(self.model.paligemma_with_expert.paligemma.parameters()).device
        img = images.to(device=device, dtype=torch.float32).permute(0, 2, 3, 1)  # pi05 preprocessing
        if tuple(img.shape[1:3]) != tuple(self.config.image_resolution):
            img = resize_with_pad_torch(img, *self.config.image_resolution)
        img = (img * 2.0 - 1.0).permute(0, 3, 1, 2)
        tokens = self.model.paligemma_with_expert.embed_image(img)
        tokens = pool_tokens(tokens.float(), self.config.memory_tokens_per_frame, self.config.memory_pool)
        return tokens.to(torch.float16)

    def _set_memory(self, feats: Tensor, times: Tensor, valid: Tensor) -> None:
        device = next(self.model.memory.parameters()).device
        tokens, mask = self.model.memory.tokens(feats.to(device), times.to(device), valid.to(device))
        self.model.memory.integration.set_context(tokens, mask)

    # ------------------------------------------------------------------ training

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        if self.uses_memory:
            if "index" not in batch:
                raise KeyError("pi05_mem training needs batch['index'] (the LeRobot dataset row)")
            if self._feature_cache is None:
                if not self.config.memory_feature_dir:
                    raise ValueError("set --policy.memory_feature_dir (see tools/build_memory_cache.py)")
                self._feature_cache = MemoryFeatureCache(self.config.memory_feature_dir)
                self._check_cache(self._feature_cache)
            index = torch.as_tensor(batch["index"])
            self._set_memory(*self._feature_cache.gather(index, self.config.memory_max_frames))
        return super().forward(batch)

    def _check_cache(self, cache: MemoryFeatureCache) -> None:
        m = cache.meta
        want = {
            "camera": self.config.memory_camera,
            "tokens_per_frame": self.config.memory_tokens_per_frame,
            "pool": self.config.memory_pool,
            "dim": image_feature_dim(self.model),
        }
        bad = {k: (m.get(k), v) for k, v in want.items() if m.get(k) != v}
        if bad:
            raise ValueError(f"memory cache {self.config.memory_feature_dir} does not match the policy: {bad}")
        if m.get("source_weights") != self.config.pretrained_path:
            logging.warning(
                "memory cache was built with weights %r, training starts from %r; the vision encoder must "
                "be the same for cached and online features",
                m.get("source_weights"),
                self.config.pretrained_path,
            )

    # ------------------------------------------------------------------ inference

    def reset(self):
        super().reset()
        self._memory_frames: list[Tensor] = []
        if self.uses_memory:
            self.model.memory.integration.set_context(None)

    @torch.no_grad()
    def observe(self, batch: dict[str, Tensor]) -> None:
        """Add the current memory-camera frame to the online buffer."""
        self._memory_frames.append(self.encode_memory_frames(batch[self.config.memory_camera]))

    def _set_online_memory(self) -> None:
        if not self._memory_frames:
            raise RuntimeError("no frames observed yet")
        frames = torch.stack(self._memory_frames, dim=1)  # (B, T, P, D)
        b, t = frames.shape[:2]
        offsets = torch.full((b,), t - 1, dtype=torch.long, device=frames.device)
        times, valid = frame_sampling_indices(offsets, self.config.memory_max_frames)
        feats = torch.gather(
            frames, 1, times[:, :, None, None].expand(-1, -1, *frames.shape[2:])
        )
        self._set_memory(feats, times, valid)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        assert not self._rtc_enabled(), "RTC is not supported for select_action, use predict_action_chunk"
        self.eval()
        if self.uses_memory:
            self.observe(batch)  # every step, also while the action queue is not empty
        if len(self._action_queue) == 0:
            actions = self._predict_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Also adds this frame to memory. Frames between calls are not seen; for chunked
        execution call observe() on those frames yourself."""
        if self.uses_memory:
            self.observe(batch)
        return self._predict_chunk(batch, **kwargs)

    def _predict_chunk(self, batch, **kwargs) -> Tensor:
        if self.uses_memory:
            self._set_online_memory()
        return super().predict_action_chunk(batch, **kwargs)

    # ------------------------------------------------------------------ loading

    def get_optim_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Memory weights may be missing (starting from a plain pi0.5 checkpoint); anything
        else missing or unexpected is an error, raised by from_pretrained below. Tied
        weights (PaliGemma's embed_tokens = lm_head) are saved once, under one name."""
        result = super().load_state_dict(state_dict, strict=False, assign=assign)
        params = self.state_dict(keep_vars=True)
        loaded_ptrs = {params[k].data_ptr() for k in state_dict if k in params}
        missing = [
            k
            for k in result.missing_keys
            if not k.startswith(MEMORY_PREFIX) and params[k].data_ptr() not in loaded_ptrs
        ]
        fresh = len(result.missing_keys) - len(missing)
        self._load_report = (missing, list(result.unexpected_keys))
        if fresh:
            logging.warning("pi05_mem: %d memory tensors not in checkpoint, initialised fresh", fresh)
        if strict and (missing or result.unexpected_keys):
            raise RuntimeError(f"state dict mismatch: missing={missing[:5]} unexpected={result.unexpected_keys[:5]}")
        return result

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *, config: PreTrainedConfig | None = None, **kwargs):
        if config is None:
            config = PreTrainedConfig.from_pretrained(pretrained_name_or_path)
        if not isinstance(config, PI05MemConfig):
            if not isinstance(config, PI05Config):
                raise TypeError(f"{pretrained_name_or_path} is a {type(config).__name__}, not pi0.5")
            fields = {f.name: getattr(config, f.name) for f in dataclasses.fields(PI05Config) if f.init}
            config = PI05MemConfig(**fields)  # a plain pi0.5 checkpoint: memory="none"
        # LeRobot's PI05Policy.from_pretrained catches load errors and returns the model
        # anyway; fail loudly instead so a fine-tune never starts from random weights.
        policy = super().from_pretrained(pretrained_name_or_path, config=config, strict=False, **kwargs)
        if policy._load_report is None:
            raise RuntimeError(f"no weights were loaded from {pretrained_name_or_path}")
        missing, unexpected = policy._load_report
        if missing or unexpected:
            raise RuntimeError(
                f"{pretrained_name_or_path}: {len(missing)} missing, {len(unexpected)} unexpected keys, "
                f"e.g. {(missing + unexpected)[:5]}"
            )
        return policy
