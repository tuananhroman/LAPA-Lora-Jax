# Troubleshooting

Common failure modes seen during LAPA-7B LoRA training, with fixes.

---

## 1. Gradient explosion (`gnorm > 100`, loss climbs)

**Symptoms**: `gnorm` rises over hundreds of steps from ~10 → ~500 → 10⁵,
loss flattens then increases, `action_acc` collapses.

**Likely cause**: effective LoRA-B LR too high *or* `clip_gradient` too
permissive. See [RECIPE.md](RECIPE.md) for the full post-mortem.

**Fix**:
1. Kill the run.
2. Resume params-only from the last clean milestone:
   `checkpoint.load: "params::/checkpoints/streaming_params_NNNN"`
3. Lower `lora.plus_ratio` (16 → 4) and `optimizer.clip_gradient` (5.0 → 1.0).
4. Do *not* change rank or base LR at the same time.

---

## 2. OOM during JIT compile or first forward

**Symptoms**: `RESOURCE_EXHAUSTED: Out of memory while trying to allocate ...`
during the first ~5 min, or at the first training step.

**Likely cause**: another process holds GPU memory, or `XLA_PYTHON_CLIENT_MEM_FRACTION`
is too high, or mesh shape doesn't match the GPU count.

**Fixes** (in order):
1. `nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader` —
   if another user holds memory, coordinate or pick different GPUs.
2. Lower `XLA_PYTHON_CLIENT_MEM_FRACTION` to `0.75`.
3. Verify `train.mesh_dim` second value × third value × fourth value matches
   the number of visible GPUs. `-1,2,1,1` requires GPU count divisible by 2.

---

## 3. `ModuleNotFoundError: No module named 'tux'`

**Cause**: launched with the wrong Python (e.g. system `python3` instead of
the lapa env).

**Fix**:
- Docker: never happens (image installs tux into system Python).
- Native: `export LAPA_PYTHON=/path/to/your/lapa/conda/bin/python3`
  before `bash scripts/run_native.sh`.

---

## 4. WandB silent / no online run created

**Causes**:
- `WANDB_API_KEY` unset → falls back to offline mode silently.
- `WANDB_MODE=disabled` set somewhere upstream.
- `logger.online: false` in the YAML overrides the env var.

**Fix**: confirm at startup banner:
```
wandb  : mode=online  project=lapa-lora-jax  run=...
```
If `mode=offline`, sync after the run: `wandb sync /workspace/outputs/.../wandb/offline-run-*`.

---

## 5. Checkpoint resume loads but loss is wrong

**Cause**: mismatched `load_checkpoint` prefix.

- `params::PATH` — weights only. Optimizer + step counter reset.
  Use this after a crash or when changing HPs.
- `trainstate::PATH` — full state (params + Adam moments + step).
  Use this for an exact, transparent continuation.

Loading a `streaming_params_NNNN` file with `trainstate::` (or vice-versa)
will silently misbehave.

---

## 6. Slow first step (10+ min before output)

**Cause**: cold XLA JIT cache. Subsequent runs reuse the on-disk cache.

**Fix**: ensure `LAPA_CACHE_DIR` is mounted to a persistent host path
(`docker-compose.yml` already does this). First run is slow; second run on
the same config is < 60 s to first step.

---

## 7. NCCL hangs at startup with multi-GPU

**Causes**: missing `--shm-size` (already 32G in compose), GPUs on different
PCIe switches with `tp > 1`, or stale `NCCL_*` env vars.

**Fix**:
- Verify topology: `nvidia-smi topo -m`. Pair FSDP GPUs on the same `PIX`/`NV*` link.
- Try `export NCCL_P2P_DISABLE=1` as a diagnostic (slower but unblocks).
- Verify `--shm-size=32g` is honoured: `docker inspect $CID | grep Shm`.
