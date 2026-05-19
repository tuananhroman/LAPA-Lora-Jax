#!/usr/bin/env bash
# =============================================================================
# setup.sh — one-time interactive setup for lapa-lora-jax
#
# Creates the local directory layout, downloads assets from HuggingFace, and
# writes a ready-to-use .env.  Safe to re-run (idempotent).
#
# Usage:
#   bash scripts/setup.sh [options]
#
# Options:
#   --non-interactive    Skip path prompts, use default directory layout (CI-friendly)
#   --skip-checkpoint    Skip the LAPA-7B checkpoint download
#   --skip-data          Skip the raw LIBERO-90 dataset download
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# =============================================================================
# Configuration — edit these if your institution mirrors are different
# =============================================================================
# HuggingFace model ID for the LAPA-7B base checkpoint (sthv2 pretrain).
# Files downloaded: params  tokenizer.model  vqgan
HF_CHECKPOINT_REPO="latent-action-pretraining/LAPA-7B-sthv2"

# HuggingFace dataset ID for raw LIBERO-90 HDF5 demos.
HF_LIBERO_REPO="LIBERO/LIBERO"
HF_LIBERO_INCLUDE="libero_90/*"     # glob pattern passed to --include

# Default host directories (relative to repo root → absolute below)
DEFAULT_CHECKPOINT_DIR="$REPO_DIR/checkpoints"
DEFAULT_DATA_DIR="$REPO_DIR/data/libero90_jsonl_v2"
DEFAULT_RAW_DATA_DIR="$REPO_DIR/data/raw"
DEFAULT_OUTPUT_DIR="$REPO_DIR/outputs"
DEFAULT_CACHE_DIR="$REPO_DIR/.cache/jax_compile"

# =============================================================================
# Parse flags
# =============================================================================
INTERACTIVE=true
DOWNLOAD_CHECKPOINT=true
DOWNLOAD_DATA=true

for arg in "$@"; do
    case "$arg" in
        --non-interactive)   INTERACTIVE=false ;;
        --skip-checkpoint)   DOWNLOAD_CHECKPOINT=false ;;
        --skip-data)         DOWNLOAD_DATA=false ;;
        -h|--help)
            sed -n '/^# Usage:/,/^# =/p' "$0" | head -n -1
            exit 0
            ;;
    esac
done

# =============================================================================
# Helpers
# =============================================================================
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "${GREEN}[setup]${NC} $*"; }
warn()    { echo -e "${YELLOW}[warn] ${NC} $*"; }
error()   { echo -e "${RED}[error]${NC} $*" >&2; }

ask_path() {
    # ask_path "prompt" default  → echoes chosen path
    local prompt="$1" default="$2"
    if ! $INTERACTIVE; then echo "$default"; return; fi
    local val
    read -rp "$(echo -e "${YELLOW}?${NC} $prompt") [$default]: " val
    echo "${val:-$default}"
}

# =============================================================================
# Banner
# =============================================================================
echo ""
echo "============================================================"
echo " LAPA LoRA-JAX — first-time setup"
echo "============================================================"
echo ""

# =============================================================================
# Prerequisite checks
# =============================================================================
info "Checking prerequisites..."
PREREQ_OK=true

if ! command -v docker &>/dev/null; then
    error "docker not found. Install Docker Engine first."
    error "  https://docs.docker.com/engine/install/"
    PREREQ_OK=false
fi

if ! docker compose version &>/dev/null 2>&1; then
    error "'docker compose' plugin not found (need Docker >= 23)."
    PREREQ_OK=false
fi

if ! docker info 2>/dev/null | grep -qi "nvidia"; then
    warn "NVIDIA Container Toolkit not detected — GPU may not work inside containers."
    warn "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/"
fi

if ! $PREREQ_OK; then
    error "Fix the above issues and re-run setup.sh."
    exit 1
fi

# Detect HuggingFace CLI
HF_CLI=""
if command -v huggingface-cli &>/dev/null; then
    HF_CLI="huggingface-cli"
elif python3 -c "import huggingface_hub" 2>/dev/null; then
    HF_CLI="python3 -m huggingface_hub"
else
    warn "huggingface-cli not found — checkpoint/data downloads will be skipped."
    warn "  Install:  pip install 'huggingface-hub[cli]'"
    DOWNLOAD_CHECKPOINT=false
    DOWNLOAD_DATA=false
fi

info "Prerequisites OK."
echo ""

# =============================================================================
# Paths
# =============================================================================
echo "--- Host directory paths (press Enter to keep defaults) ---"
CHECKPOINT_DIR=$(ask_path "LAPA-7B checkpoint dir  " "$DEFAULT_CHECKPOINT_DIR")
DATA_DIR=$(ask_path       "Preprocessed JSONL dir   " "$DEFAULT_DATA_DIR")
RAW_DATA_DIR=$(ask_path   "Raw LIBERO-90 HDF5 dir   " "$DEFAULT_RAW_DATA_DIR")
OUTPUT_DIR=$(ask_path     "Training outputs dir     " "$DEFAULT_OUTPUT_DIR")
CACHE_DIR=$(ask_path      "XLA compile cache dir    " "$DEFAULT_CACHE_DIR")
echo ""

mkdir -p "$CHECKPOINT_DIR" "$DATA_DIR" "$RAW_DATA_DIR" "$OUTPUT_DIR" "$CACHE_DIR"
info "Directories ready."
echo ""

# =============================================================================
# Generate .env
# =============================================================================
if [[ -f "$REPO_DIR/.env" ]]; then
    warn ".env already exists — skipping (delete it to regenerate)."
else
    cat > "$REPO_DIR/.env" << EOF
# Generated by scripts/setup.sh on $(date -u +"%Y-%m-%d %H:%M UTC")
# Edit any value to override defaults.

# --- Host paths --------------------------------------------------------------
# Directory containing LAPA-7B weights (params, tokenizer.model, vqgan).
LAPA_CHECKPOINT_DIR=${CHECKPOINT_DIR}

# Preprocessed training data (train.jsonl + frame images).
# Populated by:  docker compose run --rm preprocess
LAPA_DATA_DIR=${DATA_DIR}

# Raw LIBERO-90 HDF5 demos — consumed by the preprocess step only.
LAPA_RAW_DATA_DIR=${RAW_DATA_DIR}

# Saved LoRA checkpoints, train.log, WandB offline runs.
LAPA_OUTPUT_DIR=${OUTPUT_DIR}

# XLA persistent compile cache — big speedup on restarts.
LAPA_CACHE_DIR=${CACHE_DIR}

# --- GPU / compute -----------------------------------------------------------
CUDA_VISIBLE_DEVICES=0,1,2,3
XLA_PYTHON_CLIENT_MEM_FRACTION=0.82

# --- WandB -------------------------------------------------------------------
# Shared team key — replace with your own if needed.
WANDB_API_KEY=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
WANDB_PROJECT=lapa-lora-jax
WANDB_ENTITY=
WANDB_MODE=online
WANDB_RUN_NAME=lapa_lora_libero90_v2

# --- Config ------------------------------------------------------------------
# Which YAML under configs/ to load.
LAPA_CONFIG=libero90_lora_v2.yaml
EOF
    info ".env written → $REPO_DIR/.env"
    echo "       Open it and fill in WANDB_API_KEY (or set WANDB_MODE=disabled)."
fi
echo ""

# =============================================================================
# LAPA-7B checkpoint download
# =============================================================================
if $DOWNLOAD_CHECKPOINT && [[ -n "$HF_CLI" ]]; then
    EXPECTED=("params" "tokenizer.model" "vqgan")
    MISSING=()
    for f in "${EXPECTED[@]}"; do
        [[ -e "$CHECKPOINT_DIR/$f" ]] || MISSING+=("$f")
    done

    if [[ ${#MISSING[@]} -eq 0 ]]; then
        info "Checkpoint already present in $CHECKPOINT_DIR — skipping."
    else
        echo "--- LAPA-7B checkpoint (sthv2) ---"
        echo "  Repo   : https://huggingface.co/$HF_CHECKPOINT_REPO"
        echo "  Target : $CHECKPOINT_DIR"
        echo "  Missing: ${MISSING[*]}"
        echo "  Size   : ~6 GB  (downloading...)"
        if ! $HF_CLI whoami &>/dev/null 2>&1; then
            echo ""
            warn "Not logged in to HuggingFace. Running: huggingface-cli login"
            $HF_CLI login
        fi
        $HF_CLI download "$HF_CHECKPOINT_REPO" \
            --local-dir "$CHECKPOINT_DIR" \
            --local-dir-use-symlinks False
        info "Checkpoint downloaded → $CHECKPOINT_DIR"
    fi
    echo ""
fi

# =============================================================================
# LIBERO-90 raw dataset download
# =============================================================================
if $DOWNLOAD_DATA && [[ -n "$HF_CLI" ]]; then
    if [[ -n "$(ls -A "$RAW_DATA_DIR" 2>/dev/null)" ]]; then
        info "Raw LIBERO-90 data already present in $RAW_DATA_DIR — skipping."
    else
        echo "--- LIBERO-90 raw dataset ---"
        echo "  Repo   : https://huggingface.co/datasets/$HF_LIBERO_REPO"
        echo "  Target : $RAW_DATA_DIR"
        echo "  Size   : ~10 GB  (downloading...)"
        $HF_CLI download "$HF_LIBERO_REPO" \
            --repo-type dataset \
            --include "$HF_LIBERO_INCLUDE" \
            --local-dir "$RAW_DATA_DIR"
        info "LIBERO-90 raw data downloaded → $RAW_DATA_DIR"
        echo ""
        info "Run preprocessing next (GPU required, ~1-2 h):"
        echo "       docker compose run --rm preprocess"
    fi
    echo ""
fi

# =============================================================================
# Summary
# =============================================================================
DATA_READY=false
[[ -n "$(ls -A "$DATA_DIR" 2>/dev/null)" ]] && DATA_READY=true

echo "============================================================"
echo " Setup complete!  Next steps:"
echo ""
echo "  1. Edit .env — fill in WANDB_API_KEY (or set WANDB_MODE=disabled)"
echo "  2. docker compose build                      # build image (~10 min, one-time)"
if ! $DATA_READY; then
echo "  3. docker compose run --rm preprocess        # preprocess LIBERO-90 (~2 h, GPU)"
echo "  4. docker compose run --rm train dry-run     # verify config"
echo "  5. docker compose run --rm train             # start training"
else
echo "  3. docker compose run --rm train dry-run     # verify config"
echo "  4. docker compose run --rm train             # start training"
fi
echo ""
echo "  Monitor (second terminal):"
echo "    docker compose run --rm train monitor"
echo ""
echo "  Adjust training hyperparameters:"
echo "    \$EDITOR configs/libero90_lora_v2.yaml"
echo "  or override on launch:"
echo "    docker compose run --rm train train \\"
echo "        --override optimizer.lr=1e-4 --override lora.plus_ratio=2.0"
echo "============================================================"
