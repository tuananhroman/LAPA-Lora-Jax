#!/usr/bin/env bash
# Launch training from a config YAML.
#
# Usage:
#   bash scripts/run_docker.sh [configs/my_config.yaml] [-- train --override key=val ...]
#
# If a YAML path is given as the first argument it overrides LAPA_CONFIG from .env.
# Without arguments the config in .env (or the default libero90_lora_v2.yaml) is used.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"

if [[ ! -f .env ]]; then
    echo ".env not found — run 'bash scripts/setup.sh' first." >&2
    exit 2
fi

# Optional first argument: path to a YAML config (absolute or relative to repo root)
if [[ $# -gt 0 && "$1" == *.yaml ]]; then
    export LAPA_CONFIG="$(basename "$1")"
    # If the file isn't already in configs/, copy it there.
    if [[ ! -f "configs/$LAPA_CONFIG" ]]; then
        cp "$1" "configs/$LAPA_CONFIG"
        echo "[run] copied $1 → configs/$LAPA_CONFIG"
    fi
    shift
fi

echo "[run] config: ${LAPA_CONFIG:-libero90_lora_v2.yaml}"
docker compose run --rm train "$@"
