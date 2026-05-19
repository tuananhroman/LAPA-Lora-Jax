#!/usr/bin/env bash
# Convenience wrapper for `docker compose run --rm train` with .env loading.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"

if [[ ! -f .env ]]; then
    echo ".env not found. Copy .env.example to .env and fill in the paths." >&2
    exit 2
fi

docker compose run --rm train "$@"
