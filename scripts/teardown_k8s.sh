#!/bin/bash
# Remove vLLM deployments and restore Soperator workers.
# Run from your laptop.
#
# Usage: ./scripts/teardown_k8s.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
NAMESPACE="soperator"

echo "==> Removing vLLM deployments and services"
kubectl delete -f "$PROJECT_DIR/inference/k8s/base-model.yaml" --ignore-not-found
kubectl delete -f "$PROJECT_DIR/inference/k8s/lora-model.yaml" --ignore-not-found

echo "==> Waiting for vLLM pods to terminate..."
kubectl wait --for=delete pod -l app=vllm-base -n "$NAMESPACE" --timeout=60s 2>/dev/null || true
kubectl wait --for=delete pod -l app=vllm-lora -n "$NAMESPACE" --timeout=60s 2>/dev/null || true

echo "==> Restoring Soperator workers (2 replicas)"
kubectl scale statefulsets.apps.kruise.io worker -n "$NAMESPACE" --replicas=2

echo "==> Waiting for workers to be ready..."
kubectl wait --for=condition=ready pod/worker-0 pod/worker-1 -n "$NAMESPACE" --timeout=300s

echo ""
echo "=== Soperator Restored ==="
kubectl exec -n "$NAMESPACE" login-0 -- sinfo
