#!/usr/bin/env bash
# =============================================================================
# runpod_setup.sh — one-shot setup inside the RunPod container
#
# Assumes:
#   - Container image: lapa-lora-jax (built from docker/Dockerfile)
#   - RunPod Network Volume mounted at /workspace
#   - Run this script ONCE after first SSH into the pod
#
# Usage:
#   bash /opt/lapa-lora-jax/scripts/runpod_setup.sh [--skip-libero10] [--skip-preprocess]
#
# After setup, start training with:
#   LAPA_CONFIG=libero90_rslora_optionB_runpod.yaml \
#   /opt/lapa-lora-jax/scripts/entrypoint.sh train
# =============================================================================
set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn] ${NC} $*"; }
error() { echo -e "${RED}[error]${NC} $*" >&2; }
step()  { echo -e "\n${GREEN}===== $* =====${NC}"; }

SKIP_LIBERO10=false
SKIP_PREPROCESS=false
for arg in "$@"; do
    case "$arg" in
        --skip-libero10)    SKIP_LIBERO10=true ;;
        --skip-preprocess)  SKIP_PREPROCESS=true ;;
    esac
done

# =============================================================================
# 1. Symlink volume paths to expected container paths
# =============================================================================
step "1/6  Wiring /workspace volume → container paths"

mkdir -p /workspace/checkpoints /workspace/data /workspace/outputs /workspace/cache/jax_compile /workspace/raw_data

for link_target in "/checkpoints /workspace/checkpoints" \
                   "/data /workspace/data" \
                   "/raw_data /workspace/raw_data" \
                   "/cache/jax_compile /workspace/cache/jax_compile"; do
    link=$(echo $link_target | awk '{print $1}')
    target=$(echo $link_target | awk '{print $2}')
    if [[ -L "$link" ]]; then
        info "$link already symlinked — skipping."
    elif [[ -e "$link" && ! -L "$link" ]]; then
        warn "$link exists as a real directory, not overwriting."
    else
        ln -sfn "$target" "$link"
        info "  $link -> $target"
    fi
done

# /workspace/outputs is already the right path (image pre-creates it)
mkdir -p /workspace/outputs

info "Path wiring done."

# =============================================================================
# 2. Install huggingface-hub CLI if missing
# =============================================================================
step "2/6  Checking huggingface-hub"

if ! python -c "import huggingface_hub" 2>/dev/null; then
    info "Installing huggingface-hub..."
    pip install -q "huggingface-hub[cli]>=0.20"
fi
info "huggingface-hub OK."

# =============================================================================
# 3. Download LAPA-7B sthv2 checkpoint
# =============================================================================
step "3/6  LAPA-7B checkpoint  (latent-action-pretraining/LAPA-7B-sthv2)"

CKPT_DIR="/workspace/checkpoints"
EXPECTED_FILES=("params" "tokenizer.model" "vqgan")
MISSING=()
for f in "${EXPECTED_FILES[@]}"; do
    [[ -e "$CKPT_DIR/$f" ]] || MISSING+=("$f")
done

if [[ ${#MISSING[@]} -eq 0 ]]; then
    info "Checkpoint already complete — skipping download."
else
    info "Missing: ${MISSING[*]}"
    info "Downloading from HuggingFace (~6 GB)..."
    python -m huggingface_hub download \
        latent-action-pretraining/LAPA-7B-sthv2 \
        --local-dir "$CKPT_DIR" \
        --local-dir-use-symlinks False
    info "Checkpoint download complete → $CKPT_DIR"
fi

# =============================================================================
# 4. Download LIBERO-90 raw HDF5 data
# =============================================================================
step "4/6  LIBERO-90 raw HDF5  (LIBERO/LIBERO  libero_90/*)"

RAW_DIR="/workspace/raw_data"
N_HDF5=$(find "$RAW_DIR" -name "*.hdf5" 2>/dev/null | wc -l)
if [[ "$N_HDF5" -gt 0 ]]; then
    info "Found $N_HDF5 HDF5 files in $RAW_DIR — skipping download."
else
    info "Downloading LIBERO-90 HDF5 demos from HuggingFace (~10 GB)..."
    python -m huggingface_hub download \
        LIBERO/LIBERO \
        --repo-type dataset \
        --include "libero_90/*" \
        --local-dir "$RAW_DIR"
    info "LIBERO-90 download complete → $RAW_DIR"
fi

# =============================================================================
# 5. (Optional) Download LIBERO-10 for unseen evaluation
# =============================================================================
if ! $SKIP_LIBERO10; then
    step "5/6  LIBERO-10 raw HDF5  (LIBERO/LIBERO  libero_10/*)"

    LIBERO10_DIR="/workspace/raw_data/libero_10"
    N_HDF5_10=$(find "$LIBERO10_DIR" -name "*.hdf5" 2>/dev/null | wc -l)
    if [[ "$N_HDF5_10" -gt 0 ]]; then
        info "Found $N_HDF5_10 LIBERO-10 HDF5 files — skipping download."
    else
        info "Downloading LIBERO-10 HDF5 demos from HuggingFace..."
        python -m huggingface_hub download \
            LIBERO/LIBERO \
            --repo-type dataset \
            --include "libero_10/*" \
            --local-dir "$RAW_DIR"
        info "LIBERO-10 download complete."
    fi
else
    step "5/6  LIBERO-10 — skipped (--skip-libero10)"
fi

# =============================================================================
# 6. Preprocess HDF5 → JSONL  (GPU required, ~1-2 h)
# =============================================================================
step "6/6  Preprocessing LIBERO-90 → JSONL"

DATA_DIR="/workspace/data"
if [[ -f "$DATA_DIR/train.jsonl" ]]; then
    info "train.jsonl already exists in $DATA_DIR — skipping preprocessing."
    info "(Delete $DATA_DIR/train.jsonl to force re-run.)"
elif $SKIP_PREPROCESS; then
    warn "Preprocessing skipped (--skip-preprocess). Run manually:"
    warn "  python /opt/lapa-lora-jax/scripts/preprocess_libero_to_jsonl.py \\"
    warn "    --input-dir /raw_data/libero_90 --output-dir /data \\"
    warn "    --vqgan-ckpt /checkpoints/vqgan --action-vocab-size 245 \\"
    warn "    --train-demos-per-task 45 --val-demos-per-task 5"
else
    RAW_INPUT="$RAW_DIR"
    if [[ -d "$RAW_DIR/LIBERO/libero_90" ]]; then
        RAW_INPUT="$RAW_DIR/LIBERO/libero_90"
    elif [[ -d "$RAW_DIR/libero_90" ]]; then
        RAW_INPUT="$RAW_DIR/libero_90"
    fi
    info "Input  : $RAW_INPUT"
    info "Output : $DATA_DIR"
    info "VQGAN  : /checkpoints/vqgan"
    info "Starting preprocessing (this will take ~1-2 hours)..."
    python /opt/lapa-lora-jax/scripts/preprocess_libero_to_jsonl.py \
        --input-dir "$RAW_INPUT" \
        --output-dir "$DATA_DIR" \
        --vqgan-ckpt /checkpoints/vqgan \
        --action-vocab-size 245 \
        --train-demos-per-task 45 \
        --val-demos-per-task 5
    info "Preprocessing complete → $DATA_DIR"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo " RunPod setup complete!"
echo "============================================================"
echo ""
echo "  Checkpoint : $(ls /workspace/checkpoints/ | tr '\n' '  ')"
echo "  Data       : $(ls /workspace/data/ 2>/dev/null | head -5 | tr '\n' '  ')"
echo "  Outputs    : /workspace/outputs/"
echo ""
echo "  Dry-run (verify config without training):"
echo "    LAPA_CONFIG=libero90_rslora_optionB_runpod.yaml \\"
echo "    python /opt/lapa-lora-jax/scripts/launch.py \\"
echo "      /opt/lapa-lora-jax/configs/libero90_rslora_optionB_runpod.yaml --dry-run"
echo ""
echo "  Start training:"
echo "    LAPA_CONFIG=libero90_rslora_optionB_runpod.yaml \\"
echo "    WANDB_API_KEY=<your_key> \\"
echo "    /opt/lapa-lora-jax/scripts/entrypoint.sh train 2>&1 | tee /workspace/outputs/train.log"
echo ""
echo "  Keep training in background (tmux):"
echo "    tmux new -s train"
echo "    # then run the training command above"
echo "    # detach: Ctrl-B then D"
echo "============================================================"
