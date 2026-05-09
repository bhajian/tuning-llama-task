#!/bin/bash
# Sync code to cluster, setup env, and submit training job.
# Run from your laptop — no SSH needed.
#
# Usage: HF_TOKEN=hf_xxx ./scripts/deploy.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
NAMESPACE="soperator"
LOGIN_POD="login-0"
REMOTE_DIR="/root/training-task"

echo "==> Syncing code to $LOGIN_POD:$REMOTE_DIR"
kubectl exec -n "$NAMESPACE" "$LOGIN_POD" -- rm -rf "$REMOTE_DIR"
kubectl cp "$PROJECT_DIR" "$NAMESPACE/$LOGIN_POD:$REMOTE_DIR"

echo "==> Setting up environment (venv + model + dataset)"
kubectl exec -n "$NAMESPACE" "$LOGIN_POD" -- \
    bash -c "HF_TOKEN='${HF_TOKEN:-}' bash $REMOTE_DIR/scripts/setup_env.sh"

echo "==> Submitting training job"
JOB_OUTPUT=$(kubectl exec -n "$NAMESPACE" "$LOGIN_POD" -- \
    bash -c "sbatch $REMOTE_DIR/train/train.sbatch")

echo "$JOB_OUTPUT"
JOB_ID=$(echo "$JOB_OUTPUT" | grep -oE '[0-9]+$')

echo ""
echo "Job submitted. Monitor with:"
echo "  ./scripts/status.sh"
echo "  ./scripts/status.sh --logs 50"
echo "Cancel with:"
echo "  ./scripts/cancel.sh $JOB_ID"
