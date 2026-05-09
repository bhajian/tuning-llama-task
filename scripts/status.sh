#!/bin/bash
# Check job status and tail logs from your laptop.
#
# Usage: ./scripts/status.sh              # show queue + last 20 lines
#        ./scripts/status.sh --logs 100   # show queue + last 100 lines

set -euo pipefail

NAMESPACE="soperator"
LOGIN_POD="login-0"
LOG_LINES="${2:-20}"

if [ "${1:-}" = "--logs" ] && [ -n "${2:-}" ]; then
    LOG_LINES="$2"
fi

echo "=== Job Queue ==="
kubectl exec -n "$NAMESPACE" "$LOGIN_POD" -- squeue
echo ""

LATEST_OUT=$(kubectl exec -n "$NAMESPACE" "$LOGIN_POD" -- \
    bash -c 'ls -t /mnt/data/logs/train_*.out 2>/dev/null | head -1' 2>/dev/null || true)

if [ -n "$LATEST_OUT" ]; then
    LATEST_ERR="${LATEST_OUT%.out}.err"
    JOB_FILE=$(basename "$LATEST_OUT")

    echo "=== Latest Log: $JOB_FILE ==="
    kubectl exec -n "$NAMESPACE" "$LOGIN_POD" -- tail -n "$LOG_LINES" "$LATEST_OUT" 2>/dev/null || true
    echo ""

    echo "=== Stderr ==="
    kubectl exec -n "$NAMESPACE" "$LOGIN_POD" -- tail -n "$LOG_LINES" "$LATEST_ERR" 2>/dev/null || true
else
    echo "No training logs found in /mnt/data/logs/"
fi
