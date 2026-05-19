# =============================================================================
# LAPA LoRA-JAX training image
#
# Base: NVIDIA CUDA 12.3.2 + cuDNN 8.9 (matches jaxlib==0.4.23+cuda12.cudnn89)
# GPU support requires NVIDIA Container Toolkit on the host and
#   `--gpus all` / `runtime: nvidia` at run time.
# =============================================================================

FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# --- system deps ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip python3.10-venv \
        git curl ca-certificates build-essential \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3 \
    && python -m pip install --upgrade pip==24.0

WORKDIR /opt/lapa-lora-jax

# --- python deps (installed first so source edits don't bust the cache) ----
COPY requirements.txt ./requirements.txt

# JAX with CUDA 12 / cuDNN 8.9 wheels (pinned).
RUN pip install \
        "jax[cuda12]==0.4.23" \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

RUN pip install -r requirements.txt

# --- training code ----------------------------------------------------------
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

ENV PYTHONPATH=/opt/lapa-lora-jax/src \
    JAX_COMPILATION_CACHE_DIR=/cache/jax_compile \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.82 \
    XLA_FLAGS="--xla_gpu_enable_async_collectives=true" \
    WANDB_DIR=/workspace/wandb

RUN mkdir -p /cache/jax_compile /workspace/wandb /workspace/outputs

ENTRYPOINT ["/opt/lapa-lora-jax/scripts/entrypoint.sh"]
CMD ["train"]
