#!/usr/bin/env bash
# monitor.sh — live training health dashboard
#   - tails the most recent train log
#   - extracts step, loss, gnorm, action_acc lines for quick scanning
#   - prints nvidia-smi summary every refresh
#
# Usage (inside container or on host):
#   bash scripts/monitor.sh [LOG_FILE]
# If LOG_FILE is omitted, defaults to /workspace/outputs/**/train.log,
# falling back to /tmp/lapa_train.log

set -euo pipefail

LOG="${1:-}"
if [[ -z "$LOG" ]]; then
    LOG=$(find /workspace/outputs -maxdepth 4 -name 'train.log' -printf '%T@ %p\n' 2>/dev/null \
            | sort -nr | head -1 | awk '{print $2}')
fi
if [[ -z "$LOG" || ! -f "$LOG" ]]; then
    LOG="/tmp/lapa_train.log"
fi

if [[ ! -f "$LOG" ]]; then
    echo "No log file found at $LOG. Pass an explicit path:"
    echo "  bash scripts/monitor.sh /path/to/train.log"
    exit 1
fi

echo "Monitoring: $LOG"
echo "Filtering for: step | loss | gnorm | action_acc | error | OOM"
echo "Press Ctrl+C to stop."
echo "---"

stdbuf -oL -eL tail -n 200 -F "$LOG" \
    | grep --line-buffered -E 'step=|loss=|gnorm|action_acc|error|OOM|Traceback|LoRA\+'
