# LAPA LoRA-JAX

Standalone, dockerized JAX/Flax LoRA fine-tuning for **LAPA-7B**
(stage-3 LIBERO-90 by default, but the YAML is generic).

Validated **v2 recipe** that does not explode — see [docs/RECIPE.md](docs/RECIPE.md).

---

## Quick setup (new machine — 5 commands)

```bash
git clone <repo-url> lapa-lora-jax && cd lapa-lora-jax

# 1. Download checkpoint + data, write .env with paths pre-filled
bash scripts/setup.sh

# 2. Build the Docker image (one-time, ~10 min)
docker compose build

# 3. Preprocess raw LIBERO-90 HDF5 → JSONL (GPU required, ~1-2 h)
#    Skip if you already have a preprocessed data directory.
docker compose run --rm preprocess

# 4. Verify the resolved training config without launching
docker compose run --rm train dry-run

# 5. Train
docker compose run --rm train
```

`setup.sh` is interactive: it asks for directory paths (or accepts defaults),
downloads the LAPA-7B checkpoint (~6 GB) and raw LIBERO-90 data (~10 GB) from
HuggingFace, and writes a ready-to-use `.env`.  Run with
`--non-interactive` for CI / unattended machines.

---

## Why this package

`LAPA/` in the parent repo is the research codebase: many entrypoints, shell
scripts with hard-coded paths, no isolation. This package wraps the **training
path only** behind:

- a single **YAML config** ([configs/libero90_lora_v2.yaml](configs/libero90_lora_v2.yaml))
- a single **Docker image** that pins JAX 0.4.23 + CUDA 12.3 + cuDNN 8.9
- a single **`docker compose run --rm train`** command
- **WandB** integration via env vars (online/offline/disabled)

Host filesystem layout (data, checkpoints, outputs) is fully controlled by
[`.env`](.env.example). Nothing inside the image is hard-coded to a user.

---

## Hardware requirements

- ≥ 1 NVIDIA GPU with ≥ 24 GB VRAM (validated on 4× RTX 5000 Ada, 30 GB each)
- NVIDIA driver ≥ 535 (CUDA 12.x runtime compatible)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed on the host
- Linux x86_64

---

## Adjust training parameters

Edit `configs/libero90_lora_v2.yaml` directly, or override individual fields
at launch without touching any file:

```bash
docker compose run --rm train train \
    --override optimizer.lr=1e-4 \
    --override lora.plus_ratio=2.0
```

Switch to a different config:

```bash
LAPA_CONFIG=my_config.yaml docker compose run --rm train
```

---

## All available commands

```bash
# Train (default)
docker compose run --rm train

# Pre-process raw LIBERO-90 → JSONL  (GPU, ~1-2 h, one-time)
docker compose run --rm preprocess

# Dry-run: print the resolved CLI without launching training
docker compose run --rm train dry-run

# Monitor: live dashboard tailing the latest train.log
docker compose run --rm train monitor

# Interactive shell inside the container
docker compose run --rm train shell
```

---

## Quickstart (native — dev host only)

Skips Docker; uses the existing `lapa` conda env which already has JAX installed.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_API_KEY=<key> WANDB_MODE=online \
bash scripts/run_native.sh configs/libero90_lora_v2.yaml \
    2>&1 | tee /tmp/lapa_lora_native.log

# Monitor in a second terminal (auto-detects latest /tmp/lora_v*_train.log)
bash scripts/monitor_native.sh
```

---

## Expected host directory layout

Default paths (created by `setup.sh`; override in `.env`):

| Host path | `.env` var | Container mount | Mode | Contents |
| --- | --- | --- | --- | --- |
| `./checkpoints` | `LAPA_CHECKPOINT_DIR` | `/checkpoints` | ro | LAPA-7B weights (`params`, `tokenizer.model`, `vqgan`) |
| `./data/raw` | `LAPA_RAW_DATA_DIR` | `/raw_data` | ro | Raw LIBERO-90 HDF5 demos (preprocess step only) |
| `./data/libero90_jsonl_v2` | `LAPA_DATA_DIR` | `/data` | ro | Preprocessed JSONL + image tokens |
| `./outputs` | `LAPA_OUTPUT_DIR` | `/workspace/outputs` | rw | Saved checkpoints, train.log, wandb offline runs |
| `./.cache/jax_compile` | `LAPA_CACHE_DIR` | `/cache/jax_compile` | rw | XLA persistent JIT cache (large speedup on restarts) |

---

## Resume from a previous checkpoint

Two formats are supported by the underlying trainer:

```yaml
checkpoint:
  # weights only — fresh optimizer & scheduler. Use this when changing HPs
  # or recovering from a divergent run (the v2 recovery used this).
  load: "params::/checkpoints/streaming_params_2500"

  # full state — exact resume (params + Adam moments + step counter)
  load: "trainstate::/checkpoints/streaming_train_state_2500"
```

---

## WandB

| Env var | Effect |
| --- | --- |
| `WANDB_API_KEY` | Required for `WANDB_MODE=online` |
| `WANDB_PROJECT` | Project name (default `lapa-lora-jax`) |
| `WANDB_ENTITY` | Team/org slug (optional) |
| `WANDB_MODE` | `online` / `offline` / `disabled` |
| `WANDB_RUN_NAME` | Override `logger.experiment_id` from YAML |

Offline runs are written to `/workspace/outputs/.../wandb/`; sync with
`wandb sync` once you have a key.

---

## Repository layout

```
lapa-lora-jax/
├── README.md                         # this file
├── docker-compose.yml                # train + preprocess services
├── docker/Dockerfile                 # CUDA 12.3 + cuDNN 8.9 + JAX 0.4.23
├── .env.example                      # template (setup.sh fills this in)
├── pyproject.toml
├── requirements.txt                  # pinned Python deps
├── configs/
│   ├── base.yaml                     # default values (all fields documented)
│   └── libero90_lora_v2.yaml         # validated stable recipe (overrides base)
├── scripts/
│   ├── setup.sh                      # ★ first-run wizard (download + .env)
│   ├── entrypoint.sh                 # container entrypoint dispatcher
│   ├── launch.py                     # YAML → CLI translator
│   ├── preprocess_libero_to_jsonl.py # HDF5 → JSONL converter (uses VQGAN)
│   ├── monitor.sh                    # live dashboard inside container
│   ├── monitor_native.sh             # live dashboard on the dev host
│   ├── run_docker.sh                 # .env-aware compose wrapper
│   └── run_native.sh                 # bypass Docker (dev host only)
├── src/latent_pretraining/           # JAX training code (verbatim from LAPA/)
│   ├── train.py
│   ├── llama.py llama_action.py delta_llama.py delta_llama_action.py
│   ├── vision_llama.py vqgan.py
│   ├── data.py ring_attention.py
│   └── sampler_*.py
└── docs/
    ├── RECIPE.md                     # the validated v2 recipe + diagnosis
    └── TROUBLESHOOTING.md
```

---

## License & lineage

The training code under `src/latent_pretraining/` is copied verbatim from the
parent **LAPA** repository (Apache-2.0). Packaging, configs, Docker setup,
and launch scripts are this project's contribution.
