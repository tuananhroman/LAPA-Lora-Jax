#!/usr/bin/env bash
# run_native.sh — launch training directly with a conda/venv Python (no Docker)
#
# Useful for the dev host where JAX is already installed in the `lapa` env.
# Colleagues without that setup should use docker-compose instead.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_native.sh [config.yaml]
# Optional env vars:
#   LAPA_PYTHON          path to python (defaults to lapa conda env)
#   WANDB_API_KEY        enable online wandb logging
#   WANDB_MODE           online | offline | disabled
#   LAPA_OUTPUT_DIR      output root (default: /mnt/hdd/Linh/lapa_finetune_libero90_out)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

CFG="${1:-configs/libero90_lora_v2.yaml}"
[[ -f "$CFG" ]] || { echo "Config not found: $CFG"; exit 2; }

export LAPA_PYTHON="${LAPA_PYTHON:-/home/linhkastner/miniconda3/envs/lapa/bin/python3}"
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

# JAX / XLA env (mirrors validated v2 run)
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/mnt/hdd/Linh/jax_compile_cache}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.82}"
export XLA_FLAGS="${XLA_FLAGS:-}${XLA_FLAGS:+ }--xla_gpu_enable_async_collectives=true"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

# WandB defaults (override via env)
export WANDB_PROJECT="${WANDB_PROJECT:-lapa-lora-jax}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-lapa_lora_$(date +%Y%m%d_%H%M%S)}"

if [[ -n "${WANDB_API_KEY:-}" && "$WANDB_MODE" != "disabled" ]]; then
    "$LAPA_PYTHON" -m wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 || true
fi

echo "========================================================================"
echo " LAPA LoRA-JAX  (native)"
echo " config : $CFG"
echo " python : $LAPA_PYTHON"
echo " GPUs   : ${CUDA_VISIBLE_DEVICES:-(all)}"
echo " wandb  : mode=$WANDB_MODE  project=$WANDB_PROJECT  run=$WANDB_RUN_NAME"
echo "========================================================================"

exec "$LAPA_PYTHON" "$ROOT_DIR/scripts/launch.py" "$CFG" "${@:2}"
