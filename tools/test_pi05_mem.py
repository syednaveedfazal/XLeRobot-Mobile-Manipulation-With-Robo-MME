"""test_pi05_mem.py - checks for the pi0.5 memory flag (xle_policies/pi05_mem).

Uses the real pi0.5 architecture with random weights and a tiny synthetic dataset, so it
needs no downloads. "Without memory" is the same model with the memory context off,
which is exactly LeRobot's pi0.5 (the hooks then do nothing). ~10 min on one A40:

  XLE_WITH_PI=1 sbatch --output=$HOME/xle_test_pi05_mem_%j.out slurm/sim_job.sbatch \
      /isaac-sim/python.sh tools/test_pi05_mem.py

1. frame sampling and position embedding equal RoboMME's code (NumPy copies below)
2. cache tool -> training forward/backward; grads reach memory, not the vision encoder
3. Modul really acts on the training path and on the inference (denoising) path, and
   is an identity when its AdaLN projection is zero
4. memory at inference (online buffer) == memory in training (cache) for the same frames
5. save / load round trip; a plain pi0.5 checkpoint loads into a memory policy
"""

import dataclasses
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

import xle_policies  # noqa: F401
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.processor.converters import batch_to_transition, transition_to_batch
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from xle_policies.pi05_mem import PI05MemConfig, PI05MemPolicy
from xle_policies.pi05_mem.memory import MemoryFeatureCache, frame_sampling_indices, posemb_3d

REPO = Path(__file__).resolve().parents[1]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FPS, CHUNK = 30, 10
EPISODES = [50, 70]  # frames per episode
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)


# ---------------------------------------------------------------- 1. RoboMME reference copies
def robomme_even_sampling_indices(step_idx, token_budget):  # robomme:shared/data_utils.py
    if step_idx < token_budget:
        return list(range(step_idx + 1))
    return np.linspace(0, step_idx, token_budget, dtype=np.int32).tolist()


def robomme_posemb3d(pos, spatial_size, dim=768):  # robomme:shared/posemb_3d.py (jnp -> np)
    width = dim // 6
    omega = np.arange(width) / (width - 1)
    t_omega, s_omega = 1.0 / (10_000**omega), 1.0 / (1_000**omega)
    stride = 16 // spatial_size
    y, x = np.mgrid[:spatial_size, :spatial_size]
    y, x = stride * y + stride // 2, stride * x + stride // 2
    y = np.einsum("m,d->md", y.flatten(), s_omega)
    x = np.einsum("m,d->md", x.flatten(), s_omega)
    spatial = np.concatenate([np.sin(y), np.cos(y), np.sin(x), np.cos(x)], -1)
    t = np.einsum("m,d->md", pos, t_omega)
    temporal = np.concatenate([np.sin(t), np.cos(t)], -1)
    t_rep = np.repeat(temporal[:, None], spatial_size**2, 1)
    s_rep = np.repeat(spatial[None], len(pos), 0)
    return np.concatenate([t_rep, s_rep], -1)


steps = torch.arange(0, 400)
idx, valid = frame_sampling_indices(steps, 32)
same = all(idx[t][valid[t]].tolist() == robomme_even_sampling_indices(t, 32) for t in range(400))
check("frame sampling == robomme even_sampling_indices (t=0..399)", same)
pos = np.array([0, 5, 77, 1799])
err = max(
    np.abs(posemb_3d(torch.tensor(pos), g, 768).numpy() - robomme_posemb3d(pos, g)).max() for g in (2, 4, 8)
)
check("3D position embedding == robomme PosEmb3D", err < 1e-4, f"(max err {err:.1e})")

b = {"index": torch.tensor([3, 4]), "observation.state": torch.zeros(2, 3), "action": torch.zeros(2, 3)}
check("LeRobot processor converters keep batch['index']", "index" in transition_to_batch(batch_to_transition(b)))

# ---------------------------------------------------------------- tiny dataset + cache tool
root = Path(os.environ.get("XLE_DATA", tempfile.gettempdir())) / "tests" / "pi05_mem_ds"
shutil.rmtree(root, ignore_errors=True)
features = {
    "observation.images.head": {"dtype": "image", "shape": (64, 64, 3), "names": ["height", "width", "channels"]},
    "observation.images.left_wrist": {"dtype": "image", "shape": (64, 64, 3), "names": ["height", "width", "channels"]},
    "observation.state": {"dtype": "float32", "shape": (17,), "names": None},
    "action": {"dtype": "float32", "shape": (17,), "names": None},
}
rng = np.random.default_rng(0)
ds = LeRobotDataset.create("local/pi05_mem_test", FPS, features, root=root, use_videos=False)
for n in EPISODES:
    for _ in range(n):
        ds.add_frame({
            "observation.images.head": rng.integers(0, 255, (64, 64, 3), dtype=np.uint8),
            "observation.images.left_wrist": rng.integers(0, 255, (64, 64, 3), dtype=np.uint8),
            "observation.state": rng.standard_normal(17).astype(np.float32),
            "action": rng.standard_normal(17).astype(np.float32),
            "task": "remember where the cube was",
        })
    ds.save_episode()
ds.finalize()

cache_dir = root / "memory_features" / "head"
cmd = [sys.executable, str(REPO / "tools/build_memory_cache.py"), "--repo_id", "local/pi05_mem_test",
       "--root", str(root), "--pretrained_path", "", "--dtype", "float32", "--batch_size", "16",
       "--num_workers", "2"]
res = subprocess.run(cmd, capture_output=True, text=True)
check("build_memory_cache.py runs", res.returncode == 0, res.stdout[-300:] + res.stderr[-1500:])
cache = MemoryFeatureCache(cache_dir)
check("cache shape", cache.features.shape == (sum(EPISODES), 16, 2048), str(cache.features.shape))
check("cache episode starts", set(cache.episode_start.tolist()) == {0, EPISODES[0]})

# ---------------------------------------------------------------- policy
ds = LeRobotDataset("local/pi05_mem_test", root=root, delta_timestamps={"action": [i / FPS for i in range(CHUNK)]})
pf = dataset_to_policy_features(ds.meta.features)
cfg = PI05MemConfig(
    memory="framesamp_modul", dtype="float32", device=DEV, chunk_size=CHUNK, n_action_steps=CHUNK,
    num_inference_steps=3, memory_feature_dir=str(cache_dir),
)
cfg.output_features = {k: f for k, f in pf.items() if f.type is FeatureType.ACTION}
cfg.input_features = {k: f for k, f in pf.items() if k not in cfg.output_features}
torch.manual_seed(0)
mem = PI05MemPolicy(cfg).to(DEV)
integ = mem.model.memory.integration

# The cache tool used its own random vision weights; rebuild the features with these.
with torch.no_grad():
    feats = torch.cat([
        mem.encode_memory_frames(torch.stack([ds[i]["observation.images.head"] for i in range(j, min(j + 32, len(ds)))]))
        for j in range(0, len(ds), 32)
    ]).cpu().numpy()
np.save(cache_dir / "features.npy", feats)

n_mem = sum(p.numel() for p in mem.model.memory.parameters())
check("memory adds ~80M params (paper: ~80M)", 70e6 < n_mem < 100e6, f"({n_mem / 1e6:.1f}M)")
check("memory camera defaults to head", mem.config.memory_camera == "observation.images.head")

rows = [10, 49, 50, 115]  # early in episode 0, its last frame, first of episode 1, late in episode 1
batch = torch.utils.data.default_collate([ds[i] for i in rows])
batch = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in batch.items()}
tokens = torch.randint(0, 1000, (len(rows), 16), device=DEV)
batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK] = tokens, torch.ones_like(tokens, dtype=torch.bool)
rel = lambda a, b: ((a - b).norm() / b.norm()).item()  # noqa: E731


def use_memory(on: bool):
    if on:
        mem._set_memory(*mem._feature_cache.gather(batch["index"], mem.config.memory_max_frames))
    else:
        integ.set_context(None)


def raw_losses(memory_on):
    images, img_masks = mem._preprocess_images(batch)
    actions = mem.prepare_action(batch)
    g = torch.Generator(device=DEV).manual_seed(1)
    noise = torch.randn(actions.shape, generator=g, device=DEV)
    time = torch.rand(actions.shape[0], generator=g, device=DEV) * 0.9 + 0.05
    use_memory(memory_on)
    return mem.model.forward(images, img_masks, tokens, batch[OBS_LANGUAGE_ATTENTION_MASK], actions, noise, time)


def sample(memory_on):
    images, img_masks = mem._preprocess_images(batch)
    g = torch.Generator(device=DEV).manual_seed(2)
    noise = torch.randn(len(rows), CHUNK, mem.config.max_action_dim, generator=g, device=DEV)
    use_memory(memory_on)
    return mem.model.sample_actions(images, img_masks, tokens, batch[OBS_LANGUAGE_ATTENTION_MASK], noise=noise)


# ---------------------------------------------------------------- 2. training step
mem.train()
loss, info = mem.forward(batch)
loss.backward()
grad = lambda m: sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)  # noqa: E731
pg = mem.model.paligemma_with_expert.paligemma.model
check("training loss finite", torch.isfinite(loss).item(), f"(loss {loss.item():.3f})")
check("grads reach memory encoder and every Modul layer",
      grad(mem.model.memory.encoder) > 0 and all(grad(m) > 0 for m in integ.layers))
check("vision encoder + projector frozen (no grads)",
      all(p.grad is None for p in [*pg.vision_tower.parameters(), *pg.multi_modal_projector.parameters()]))
check("VLM and action expert still train", grad(pg.language_model) > 0 and grad(mem.model.paligemma_with_expert.gemma_expert) > 0)
check("optimizer gets only trainable params", all(p.requires_grad for p in mem.get_optim_params()))
mem.zero_grad(set_to_none=True)

mem.model.gradient_checkpointing_enable()
loss_gc, _ = mem.forward(batch)
loss_gc.backward()
check("works with gradient checkpointing", grad(mem.model.memory.encoder) > 0)
mem.model.gradient_checkpointing_disable()
mem.zero_grad(set_to_none=True)
torch.cuda.empty_cache()

# ---------------------------------------------------------------- 3. Modul acts on both paths
mem.eval()
with torch.no_grad():
    l_base, a_base = raw_losses(False), sample(False)
    d = rel(raw_losses(True), l_base)
    check("at init the memory policy is close to pi0.5", d < 0.5, f"(rel {d:.2e})")
    saved = {k: v.clone() for k, v in mem.model.memory.state_dict().items()}
    for layer in integ.layers:
        layer.mod.weight.zero_()
        layer.mod.bias.zero_()
    d = rel(raw_losses(True), l_base)
    check("zero AdaLN projection -> identical to pi0.5 (training path)", d < 1e-4, f"(rel {d:.1e})")
    d = rel(sample(True), a_base)
    check("zero AdaLN projection -> identical to pi0.5 (inference path)", d < 1e-4, f"(rel {d:.1e})")
    for layer in integ.layers:
        layer.mod.weight.normal_(std=0.05)
    d = rel(raw_losses(True), l_base)
    check("Modul changes the training loss", d > 0.05, f"(rel {d:.2f})")
    d = rel(sample(True), a_base)
    check("Modul changes the sampled actions", d > 0.01, f"(rel {d:.2f})")
    m1 = sample(True)
    real = mem._feature_cache.features
    mem._feature_cache.features = np.zeros_like(real)  # same model, different memory
    d = rel(sample(True), m1)
    check("different memory -> different actions", d > 1e-3, f"(rel {d:.2e})")
    mem._feature_cache.features = real
    mem.model.memory.load_state_dict(saved)

# ---------------------------------------------------------------- 4. online buffer == cache
# Episode 1 played step by step through select_action; at its frame 40 the memory tokens
# must equal what training builds for the same frame from the cache.
mem.reset()
ep1 = EPISODES[0]
with torch.no_grad():
    for f in range(41):
        item = ds[ep1 + f]
        step = {k: (v[None].to(DEV) if torch.is_tensor(v) else [v]) for k, v in item.items()}
        step[OBS_LANGUAGE_TOKENS] = tokens[:1]
        step[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones_like(tokens[:1], dtype=torch.bool)
        act = mem.select_action(step)
    online_tokens, online_mask = integ._context
    train_tokens, train_mask = mem.model.memory.tokens(
        *[t.to(DEV) for t in mem._feature_cache.gather(torch.tensor([ep1 + 40]), mem.config.memory_max_frames)]
    )
check("select_action returns finite actions", torch.isfinite(act).all().item(), str(tuple(act.shape)))
check("online buffer has one frame per step", len(mem._memory_frames) == 41)
d = (online_tokens - train_tokens).abs().max().item()
check("online memory == training memory (same frames)", torch.equal(online_mask, train_mask) and d < 2e-2,
      f"(max diff {d:.1e}, frames used {int(online_mask.sum()) // 16})")
mem.reset()
check("reset clears the buffer", len(mem._memory_frames) == 0 and integ._context is None)

# ---------------------------------------------------------------- 5. save / load (node-local /tmp)
if shutil.disk_usage(tempfile.gettempdir()).free < 40e9:
    print("[SKIP] save/load: less than 40 GB free in", tempfile.gettempdir())
else:
    ref = {k: v.cpu() for k, v in mem.state_dict().items()}
    with tempfile.TemporaryDirectory() as d:
        mem.save_pretrained(Path(d) / "mem")
        loaded = PI05MemPolicy.from_pretrained(Path(d) / "mem")
        same = all(torch.equal(ref[k], v.cpu()) for k, v in loaded.state_dict().items())
        check("save/load keeps type, flag and weights",
              loaded.config.type == "pi05_mem" and loaded.config.memory == "framesamp_modul" and same)
        del loaded
        torch.cuda.empty_cache()

        # a plain LeRobot pi0.5 checkpoint: config type "pi05", no memory weights
        memory = mem.model.memory
        mem.model.memory = None
        mem.save_pretrained(Path(d) / "pi05")
        mem.model.memory = memory
        PI05Config(**{f.name: getattr(cfg, f.name) for f in dataclasses.fields(PI05Config) if f.init})._save_pretrained(Path(d) / "pi05")
        plain = PI05MemPolicy.from_pretrained(Path(d) / "pi05")
        check("pi0.5 checkpoint loads as memory=none", plain.config.memory == "none" and not plain.uses_memory)
        del plain
        torch.cuda.empty_cache()
        with_mem = PI05MemPolicy.from_pretrained(Path(d) / "pi05", config=cfg)
        same = all(torch.equal(ref[k], v.cpu()) for k, v in with_mem.state_dict().items() if not k.startswith("model.memory."))
        check("pi0.5 checkpoint -> memory policy (pi0.5 weights loaded, memory fresh)", with_mem.uses_memory and same)

print("\nALL PASSED" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
