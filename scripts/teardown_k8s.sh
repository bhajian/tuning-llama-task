#!/bin/bash
# Remove vLLM deployments, restore Soperator workers, resume FluxCD.
# Run from your laptop.
#
# Usage: ./scripts/teardown_k8s.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SOPERATOR_NS="soperator"
INFERENCE_NS="inference"
HELMRELEASE="flux-system-soperator-fluxcd-slurm-cluster"

echo "==> Removing vLLM deployments and services"
kubectl delete -f "$PROJECT_DIR/inference/k8s/lora-model.yaml" --ignore-not-found
kubectl delete -f "$PROJECT_DIR/inference/k8s/base-model.yaml" --ignore-not-found

echo "==> Waiting for vLLM pods to terminate..."
kubectl wait --for=delete pod -l 'app in (vllm-base, vllm-lora)' -n "$INFERENCE_NS" --timeout=60s 2>/dev/null || true

echo "==> Restoring Soperator workers (2 replicas)"
kubectl scale statefulsets.apps.kruise.io worker -n "$SOPERATOR_NS" --replicas=2

echo "==> Resuming FluxCD reconciliation"
kubectl patch helmrelease "$HELMRELEASE" -n flux-system \
    --type merge -p '{"spec":{"suspend":false}}'

echo "==> Waiting for workers to be ready..."
kubectl wait --for=condition=ready pod/worker-0 pod/worker-1 -n "$SOPERATOR_NS" --timeout=300s

echo ""
echo "=== Soperator Restored ==="
kubectl exec -n "$SOPERATOR_NS" login-0 -- sinfo
