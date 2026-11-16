"""FrameSamp + Modul memory for pi0.5, ported from RoboMME's JAX code to PyTorch.

Reference: Dai et al., "RoboMME: Benchmarking and Understanding Memory for Robotic
Generalist Policies", ICML 2026, and github.com/RoboMME/robomme_policy_learning
@ac846a9 (files named below as `robomme:<path>`).

FrameSamp (representation)
  Every frame of the memory camera goes through pi0.5's own frozen SigLIP + projector,
  and its 16x16 tokens are pooled to 4x4. At step t, N = budget / tokens_per_frame
  frames are taken evenly from 0..t (all of them while t < N). Their tokens, plus a
  3D sin-cos position (time, row, col), are projected to the action expert's width.

Modul (integration)
  In every action-expert layer, just before the feed-forward block, the action tokens
  cross-attend to the memory tokens; the result is mapped to a scale and shift that
  modulate the RMS-normalised action features (AdaLN). The VLM is untouched. It starts
  close to an identity (scale ~ 0, shift ~ 0), so training starts from plain pi0.5.
"""

import math
from functools import partial
from pathlib import Path

import json
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

SIGLIP_GRID = 16  # 224 px / 14 px patches
_BIG_NEG = -2.3819763e38  # robomme / openpi attention mask value


# --------------------------------------------------------------------------- FrameSamp


def frame_sampling_indices(offsets: Tensor, max_frames: int) -> tuple[Tensor, Tensor]:
    """Frames to remember at each current frame offset (robomme:shared/data_utils.py
    even_sampling_indices): all of 0..t while t < N, else N evenly spaced, both ends kept.

    offsets: (B,) current frame index within its episode.
    Returns (indices (B, N) long, valid (B, N) bool). Invalid slots repeat t and are masked.
    """
    t = offsets.long()[:, None]
    j = torch.arange(max_frames, device=offsets.device)[None, :]
    if max_frames == 1:
        return t.clone(), torch.ones_like(t, dtype=torch.bool)
    spaced = (j * t) // (max_frames - 1)  # == np.linspace(0, t, N, dtype=int32) for t >= N-1
    short = t < max_frames
    idx = torch.where(short, j, spaced)
    valid = ~short | (j <= t)
    return torch.minimum(idx, t), valid


def posemb_3d(times: Tensor, grid: int, dim: int) -> Tensor:
    """3D sin-cos position embedding (robomme:shared/posemb_3d.py PosEmb3D).

    times: (...) frame index. Returns (..., grid*grid, dim): [time sin/cos | row sin/cos | col sin/cos].
    """
    width = dim // 6
    device = times.device
    omega = torch.arange(width, dtype=torch.float32, device=device) / (width - 1)
    t_omega = 1.0 / (10_000.0**omega)
    s_omega = 1.0 / (1_000.0**omega)

    t = times.float()[..., None] * t_omega
    temporal = torch.cat([torch.sin(t), torch.cos(t)], dim=-1)  # (..., 2w)

    stride = SIGLIP_GRID // grid  # positions of pooled cells on the 16x16 patch grid
    coords = torch.arange(grid, device=device, dtype=torch.float32) * stride + stride // 2
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    y = y.flatten()[:, None] * s_omega
    x = x.flatten()[:, None] * s_omega
    spatial = torch.cat([torch.sin(y), torch.cos(y), torch.sin(x), torch.cos(x)], dim=-1)  # (g*g, 4w)

    temporal = temporal[..., None, :].expand(*times.shape, grid * grid, 2 * width)
    spatial = spatial.expand(*times.shape, grid * grid, 4 * width)
    return torch.cat([temporal, spatial], dim=-1)


def pool_tokens(tokens: Tensor, tokens_per_frame: int, mode: str = "mean") -> Tensor:
    """(B, 256, D) SigLIP tokens -> (B, tokens_per_frame, D), pooled on the 16x16 grid."""
    b, p, d = tokens.shape
    side = math.isqrt(p)
    out = math.isqrt(tokens_per_frame)
    if side == out:
        return tokens
    grid = tokens.transpose(1, 2).reshape(b, d, side, side)
    k = side // out
    pooled = F.avg_pool2d(grid, k) if mode == "mean" else F.max_pool2d(grid, k)
    return pooled.flatten(2).transpose(1, 2)


class FrameSampEncoder(nn.Module):
    """Pooled frame tokens (+ position) -> memory tokens (robomme:models/representation/mem_encoder.py)."""

    def __init__(self, img_dim: int, out_dim: int, pos_dim: int, use_pos_emb: bool):
        super().__init__()
        self.pos_proj = nn.Linear(pos_dim, pos_dim) if use_pos_emb else None
        self.proj = nn.Linear(img_dim + (pos_dim if use_pos_emb else 0), out_dim)
        for lin in (self.pos_proj, self.proj):
            if lin is not None:
                nn.init.normal_(lin.weight, std=0.02)
                nn.init.zeros_(lin.bias)

    def forward(self, img: Tensor, pos: Tensor | None) -> Tensor:
        x = img
        if self.pos_proj is not None:
            x = torch.cat([x, F.silu(self.pos_proj(pos))], dim=-1)
        return self.proj(x)


# --------------------------------------------------------------------------- Modul


def _rms(x: Tensor) -> Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


def _rope(x: Tensor, positions: Tensor) -> Tensor:
    """openpi gemma _apply_rope. x: (B, L, H, D), positions: (L,)."""
    half = x.shape[-1] // 2
    freq = (2.0 / x.shape[-1]) * torch.arange(half, dtype=torch.float32, device=x.device)
    radians = positions.float()[:, None] / (10_000.0**freq)[None, :]  # (L, half)
    sin, cos = torch.sin(radians)[None, :, None, :], torch.cos(radians)[None, :, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def _lecun_normal_(weight: Tensor) -> None:
    """flax lecun_normal: truncated normal, variance 1/fan_in."""
    std = math.sqrt(1.0 / weight.shape[1]) / 0.87962566103423978
    nn.init.trunc_normal_(weight, std=std, a=-2 * std, b=2 * std)


class ModulLayer(nn.Module):
    """robomme:models/integration/history_gemma.py MemoryAttention + MemoryRMSNorm(cond)."""

    def __init__(self, width: int, head_dim: int, num_heads: int, num_kv_heads: int = 1):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = num_heads, num_kv_heads, head_dim
        self.norm_scale = nn.Parameter(torch.zeros(width))  # one norm shared by actions and memory
        self.q = nn.Linear(width, num_heads * head_dim, bias=False)
        self.kv = nn.Linear(width, 2 * num_kv_heads * head_dim, bias=False)
        self.o = nn.Linear(num_heads * head_dim, width, bias=False)
        self.mod = nn.Linear(width, 2 * width)  # -> AdaLN scale, shift
        for lin in (self.q, self.kv, self.o):
            _lecun_normal_(lin.weight)
        nn.init.normal_(self.mod.weight, std=0.002)  # "small randomness instead of pure zeros"
        nn.init.zeros_(self.mod.bias)

    def forward(self, x: Tensor, mem: Tensor, mem_mask: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        b, t, _ = x.shape
        s = mem.shape[1]
        h, k, d = self.num_heads, self.num_kv_heads, self.head_dim

        q = self.q(_rms(x) * (1 + self.norm_scale)).view(b, t, h, d)
        kv = self.kv(_rms(mem) * (1 + self.norm_scale)).view(b, s, 2, k, d)
        key, val = kv[:, :, 0], kv[:, :, 1]
        # memory sits at positions 0..S-1, the action tokens right after it
        q = _rope(q, torch.arange(s, s + t, device=x.device)) * d**-0.5
        key = _rope(key, torch.arange(s, device=x.device))

        q = q.view(b, t, k, h // k, d)
        logits = torch.einsum("btkgd,bskd->bkgts", q, key)
        logits = logits.masked_fill(~mem_mask[:, None, None, None, :], _BIG_NEG)
        probs = torch.softmax(logits, dim=-1)
        attended = torch.einsum("bkgts,bskd->btkgd", probs, val).reshape(b, t, h * d)

        scale, shift = self.mod(self.o(attended)).chunk(2, dim=-1)
        return (_rms(x) * (1 + scale) + shift).to(dtype)


class ModulIntegration(nn.Module):
    """One ModulLayer per action-expert layer, applied to the input of that layer's
    pre-feed-forward norm (`post_attention_layernorm` in HF Gemma) by a forward pre-hook.

    The hook covers both of LeRobot's code paths: the joint VLM+expert pass used in
    training (modeling_pi05.compute_layer_complete) and the expert-only denoising pass
    at inference (HF GemmaDecoderLayer). The residual stream is left as it was, exactly
    as in RoboMME: only the feed-forward input is modulated.
    """

    def __init__(self, expert_layers: nn.ModuleList, width: int, head_dim: int, num_heads: int):
        super().__init__()
        self.layers = nn.ModuleList(ModulLayer(width, head_dim, num_heads) for _ in expert_layers)
        self._context: tuple[Tensor, Tensor] | None = None
        for i, layer in enumerate(expert_layers):
            layer.post_attention_layernorm.register_forward_pre_hook(partial(self._hook, i), with_kwargs=True)

    def set_context(self, tokens: Tensor | None, mask: Tensor | None = None) -> None:
        # Kept until the next call: gradient checkpointing re-runs the hooks during backward.
        self._context = None if tokens is None else (tokens, mask)

    def _hook(self, i, module, args, kwargs):
        if self._context is None:
            return None
        tokens, mask = self._context
        return (self.layers[i](args[0], tokens, mask), *args[1:]), kwargs


class FrameSampModul(nn.Module):
    def __init__(self, config, expert_layers: nn.ModuleList, img_dim: int, expert_width: int, head_dim: int):
        super().__init__()
        self.config = config
        self.encoder = FrameSampEncoder(
            img_dim, expert_width, config.memory_pos_emb_dim, config.memory_use_pos_emb
        )
        self.integration = ModulIntegration(expert_layers, expert_width, head_dim, config.memory_num_heads)

    def tokens(self, feats: Tensor, times: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        """feats (B, N, P, D) pooled SigLIP features, times (B, N) frame index, valid (B, N)
        -> memory tokens (B, N*P, width), mask (B, N*P)."""
        b, n, p, _ = feats.shape
        pos = None
        if self.config.memory_use_pos_emb:
            pos = posemb_3d(times, math.isqrt(p), self.config.memory_pos_emb_dim)
        feats = feats.float()
        # padded frames are zero, as in robomme right_padding_token_emb
        feats = feats * valid[..., None, None]
        if pos is not None:
            pos = (pos * valid[..., None, None]).reshape(b, n * p, -1)
        tokens = self.encoder(feats.reshape(b, n * p, -1), pos)
        mask = valid[:, :, None].expand(b, n, p).reshape(b, n * p)
        return tokens, mask


def image_feature_dim(pi05_model: nn.Module) -> int:
    """Size of pi0.5's per-patch image features (SigLIP + projector): 2048."""
    return pi05_model.paligemma_with_expert.paligemma.config.vision_config.projection_dim


def build_memory(config, pi05_model: nn.Module) -> nn.Module | None:
    """The memory module for config.memory, attached to the pi0.5 model, or None."""
    if config.memory == "none":
        return None
    pwe = pi05_model.paligemma_with_expert
    expert = pwe.gemma_expert.model
    img_dim = image_feature_dim(pi05_model)
    width = expert.config.hidden_size
    head_dim = expert.config.head_dim
    if config.memory == "framesamp_modul":
        return FrameSampModul(config, expert.layers, img_dim, width, head_dim)
    raise ValueError(f"no builder for memory={config.memory!r}")


# --------------------------------------------------------------------------- feature cache


class MemoryFeatureCache:
    """Pooled SigLIP features of every frame of one dataset camera, row = dataset `index`.

    Written by tools/build_memory_cache.py:
      features.npy        float16 (num_frames, tokens_per_frame, dim), memory-mapped
      episode_start.npy   int64   (num_frames,) index of the first frame of each frame's episode
      meta.json
    """

    def __init__(self, path: str | Path):
        path = Path(path)
        if not (path / "features.npy").exists():
            raise FileNotFoundError(
                f"no memory feature cache in {path}; build it with tools/build_memory_cache.py"
            )
        self.meta = json.loads((path / "meta.json").read_text())
        self.features = np.load(path / "features.npy", mmap_mode="r")
        self.episode_start = np.load(path / "episode_start.npy")

    def gather(self, index: Tensor, max_frames: int) -> tuple[Tensor, Tensor, Tensor]:
        """index (B,) dataset rows -> feats (B, N, P, D) fp16, frame times (B, N), valid (B, N)."""
        index = index.detach().cpu().long().reshape(-1)
        if int(index.max()) >= len(self.episode_start):
            raise IndexError(
                f"dataset index {int(index.max())} beyond cache ({len(self.episode_start)} frames); rebuild it"
            )
        start = torch.from_numpy(self.episode_start[index.numpy()])
        times, valid = frame_sampling_indices(index - start, max_frames)
        rows = (start[:, None] + times).numpy()
        feats = torch.from_numpy(np.ascontiguousarray(self.features[rows.reshape(-1)]))
        return feats.view(*rows.shape, *feats.shape[1:]), times, valid
