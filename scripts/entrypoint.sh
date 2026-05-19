#!/usr/bin/env bash
# Container entrypoint. Dispatches to one of:
#   train     — run training with the YAML config from $LAPA_CONFIG
#   monitor   — tail the latest training log
#   shell     — open an interactive bash shell
#   <other>   — exec the given command as-is

set -euo pipefail

cd /opt/lapa-lora-jax

# WandB login if a key was provided (silent if already logged in).
if [[ -n "${WANDB_API_KEY:-}" && "${WANDB_MODE:-online}" != "disabled" ]]; then
    wandb login --relogin --host="${WANDB_BASE_URL:-https://api.wandb.ai}" "${WANDB_API_KEY}" >/dev/null 2>&1 || true
fi

CMD="${1:-train}"
shift || true

case "$CMD" in
    train)
        CFG="${LAPA_CONFIG:-libero90_lora_v2.yaml}"
        CFG_PATH="/opt/lapa-lora-jax/configs/${CFG}"
        if [[ ! -f "$CFG_PATH" ]]; then
            echo "Config not found: $CFG_PATH" >&2
            echo "Available:" >&2
            ls /opt/lapa-lora-jax/configs/ >&2
            exit 2
        fi
        exec python /opt/lapa-lora-jax/scripts/launch.py "$CFG_PATH" "$@"
        ;;
    preprocess)
        # Preprocess raw LIBERO-90 HDF5 files → train.jsonl + val.jsonl.
        # Expects:
        #   /checkpoints/vqgan  — LAPA VQGAN checkpoint (from LAPA_CHECKPOINT_DIR)
        #   /raw_data           — raw HDF5 files dir   (from LAPA_RAW_DATA_DIR)
        #   /processed_data     — output dir           (from LAPA_DATA_DIR, writable)
        #
        # The script auto-detects the HF download sub-directory structure:
        #   /raw_data/libero_90/*.hdf5   OR
        #   /raw_data/LIBERO/libero_90/*.hdf5
        RAW="/raw_data"
        if [[ -d "$RAW/LIBERO/libero_90" ]]; then
            RAW="$RAW/LIBERO/libero_90"
        elif [[ -d "$RAW/libero_90" ]]; then
            RAW="$RAW/libero_90"
        fi
        echo "[preprocess] input  : $RAW"
        echo "[preprocess] output : /processed_data"
        echo "[preprocess] vqgan  : /checkpoints/vqgan"
        exec python /opt/lapa-lora-jax/scripts/preprocess_libero_to_jsonl.py \
            --input-dir "$RAW" \
            --output-dir /processed_data \
            --vqgan-ckpt /checkpoints/vqgan \
            --action-vocab-size 245 \
            --train-demos-per-task 45 \
            --val-demos-per-task 5 \
            --vqgan-batch 64 \
            "$@"
        ;;
    monitor)
        exec bash /opt/lapa-lora-jax/scripts/monitor.sh "$@"
        ;;
    dry-run)
        CFG="${LAPA_CONFIG:-libero90_lora_v2.yaml}"
        exec python /opt/lapa-lora-jax/scripts/launch.py \
            "/opt/lapa-lora-jax/configs/${CFG}" --dry-run "$@"
        ;;
    shell|bash)
        exec /bin/bash "$@"
        ;;
    *)
        exec "$CMD" "$@"
        ;;
esac
