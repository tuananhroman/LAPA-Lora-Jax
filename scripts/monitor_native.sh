#!/usr/bin/env bash
# monitor_native.sh — live training dashboard for the dev host (no Docker)
#
# Shows: step, loss, gnorm, action_acc, errors.
# Also prints nvidia-smi GPU summary on each print.
#
# Usage:
#   bash scripts/monitor_native.sh                    # auto-detect latest log
#   bash scripts/monitor_native.sh /tmp/lora_v2_train.log
#   LOG=/tmp/lora_v2_train.log bash scripts/monitor_native.sh
#
# Typical log location: /tmp/lora_v*_train.log (set in run_native.sh / v2 script)

set -euo pipefail

LOG="${1:-${LOG:-}}"

# Auto-detect most recent lora train log under /tmp
if [[ -z "$LOG" ]]; then
    LOG=$(ls -t /tmp/lora_v*_train.log /tmp/lapa_*_train.log 2>/dev/null | head -1 || true)
fi

if [[ -z "$LOG" || ! -f "$LOG" ]]; then
    echo "No log found. Pass an explicit path or set LOG=/path/to/train.log"
    echo "Available in /tmp:" && ls /tmp/*.log 2>/dev/null || echo "  (none)"
    exit 1
fi

echo "========================================================================"
echo " LAPA training monitor (native)"
echo " log    : $LOG"
echo " filter : step | loss | gnorm | action_acc | error | OOM"
echo " Ctrl+C to stop"
echo "========================================================================"

# Print GPU summary header every N matching lines
N=50
count=0

stdbuf -oL -eL tail -n 200 -F "$LOG" \
    | grep --line-buffered -E 'step=|loss=|gnorm|action_acc|[Ee]rror|OOM|Traceback|LoRA\+' \
    | while IFS= read -r line; do
        echo "$line"
        count=$((count + 1))
        if (( count % N == 0 )); then
            echo ""
            echo "--- GPU @ $(date '+%H:%M:%S') ---"
            nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu \
                       --format=csv,noheader 2>/dev/null || true
            echo "------"
        fi
    done
