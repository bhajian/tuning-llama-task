#!/bin/bash
# Cancel Slurm jobs from your laptop.
#
# Usage: ./scripts/cancel.sh         # cancel all jobs
#        ./scripts/cancel.sh 42      # cancel job 42

set -euo pipefail

NAMESPACE="soperator"
LOGIN_POD="login-0"

if [ -n "${1:-}" ]; then
    echo "Cancelling job $1..."
    kubectl exec -n "$NAMESPACE" "$LOGIN_POD" -- scancel "$1"
else
    echo "Cancelling all jobs..."
    kubectl exec -n "$NAMESPACE" "$LOGIN_POD" -- bash -c 'scancel --user=root 2>/dev/null || scancel -u root 2>/dev/null || echo "No jobs to cancel"'
fi

echo ""
echo "=== Remaining Jobs ==="
kubectl exec -n "$NAMESPACE" "$LOGIN_POD" -- squeue
