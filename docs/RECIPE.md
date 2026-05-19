# Validated v2 recipe — LAPA-7B LoRA on LIBERO-90

This recipe is the post-mortem fix of the **v1 gradient explosion** observed
in May 2026. Use it as the starting point; only deviate after reading the
diagnosis below.

## Recipe (all values in [configs/libero90_lora_v2.yaml](../configs/libero90_lora_v2.yaml))

| Group | Field | Value | Notes |
| --- | --- | --- | --- |
| Model | base | LAPA-7B-Sthv2 (HF) | params-only load |
| Mesh | `mesh_dim` | `-1,2,1,1` | dp=auto, fsdp=2, tp=1, sp=1 — pair FSDP on same PCIe switch |
| Optim | `lr` | 2e-4 | base (lora_A) LR |
| Optim | `end_lr` | 1e-5 | cosine target |
| Optim | `lr_warmup_steps` | 500 | |
| Optim | `lr_decay_steps` | 25 000 | |
| Optim | `weight_decay` | 0.01 | |
| Optim | `clip_gradient` | **1.0** | v1 was 5.0 — too permissive |
| Optim | `bf16_momentum` | true | saves ~7 GB on 7B model |
| Optim | `accumulate_gradient_steps` | 4 | effective batch = 32 × 4 = 128 |
| LoRA | `rank` | 64 | |
| LoRA | `alpha` | 64 | with rsLoRA → scaling = α/√r = 8.0 |
| LoRA | `use_rslora` | true | |
| LoRA | `plus_ratio` | **4.0** | v1 was 16.0 — too aggressive |
| LoRA | `heads_lr_multiplier` | 2.0 | action / vision / delta heads |
| Train | `total_steps` | 25 000 | ~14.7 h on 4× RTX 5000 Ada |

## What changed vs v1

Only two values, for clean attribution:

1. `lora.plus_ratio: 16.0 → 4.0`
2. `optimizer.clip_gradient: 5.0 → 1.0`

Everything else is identical to v1 (same LR, same rank, same schedule, same
mesh, same dataset).

## Diagnosis of the v1 explosion

```
Steps 0    – 4520 :  stable; loss 5.58 → 3.57; gnorm 1–30 (clipped at 5.0)
Step  3710        :  first warning, gnorm = 272 (precursor)
Step  4530        :  explosion begins, gnorm = 640
Step  4720        :  peak, gnorm = 524,288
Step  5340        :  collapsed; loss = 5.03; action_acc = 0.08 (worse than init)
```

**Root cause.** With `plus_ratio=16` and `base_lr=2e-4`, the effective LR on
`lora_B` is `3.2e-3`. The gradient through `B` is `∇W · Aᵀ · scale`, so as
`A` accumulated structure over ~3 700 steps the B-gradient magnitude grew.
Adam's second-moment estimator was unable to track this growth at clip=5.0
(gnorms of 30–150 leaked through, corrupting moments). The runaway then
cascaded.

**Why these two fixes.**

- `plus_ratio 16 → 4`: the LoRA+ paper (arXiv 2402.12354) shows `λ=16` is
  safe **only when paired with the smaller LRs typical of pre-training**
  (≤ 5e-5). HF's PEFT example uses `λ=16` with `lr=5e-5`, which is 4× lower
  than ours. Dropping to `λ=4` gives `B-lr = 8e-4`, still 4× base — the
  paper's well-tested setting.
- `clip 5.0 → 1.0`: 1.0 is the **HF Trainer / accelerate / deepspeed default**
  for LLM fine-tuning and the only value that catches early-warning spikes
  before they corrupt Adam moments.

## Recovery procedure (already executed)

```text
1. kill the diverged process (PID 5590)
2. resume params-only from streaming_params_2500 (last clean checkpoint)
3. apply the two fixes above
4. fresh optimizer state, fresh LR schedule (warmup re-applied)
```

The params-only resume is critical: re-using the v1 Adam moments would
re-introduce the corrupted state.

## Acceptance criteria for any future run

- `gnorm < 5.0` sustained throughout (occasional spike up to ~10 OK)
- `loss < 3.86` reached by step 1 500 (matches v1 step-2500 baseline)
- `action_acc` monotonically increasing in the first 2 000 steps

## If v2 still spikes (very unlikely)

Next lever, **one at a time**, in order:

1. `optimizer.lr: 2e-4 → 1e-4` (halve base LR)
2. `lora.plus_ratio: 4.0 → 2.0`
3. `lora.rank: 64 → 32`

Do **not** change rank, base LR, and LoRA+ ratio simultaneously — you lose
attribution.
