#!/bin/bash
# Scale down Soperator workers, deploy vLLM serving on K8s.
# Run from your laptop.
#
# Usage: ./scripts/serve_k8s.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
NAMESPACE="soperator"

echo "==> Scaling down Soperator workers to free GPUs"
kubectl scale statefulsets.apps.kruise.io worker -n "$NAMESPACE" --replicas=0

echo "==> Waiting for worker pods to terminate..."
kubectl wait --for=delete pod/worker-0 pod/worker-1 -n "$NAMESPACE" --timeout=120s 2>/dev/null || true

echo "==> Verifying GPUs are available"
GPU_COUNT=$(kubectl get nodes -o jsonpath='{range .items[*]}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}' | awk '{s+=$1} END{print s}')
echo "    Available GPUs: $GPU_COUNT"

echo "==> Deploying vLLM base model server"
kubectl apply -f "$PROJECT_DIR/inference/k8s/base-model.yaml"

echo "==> Deploying vLLM LoRA model server"
kubectl apply -f "$PROJECT_DIR/inference/k8s/lora-model.yaml"

echo "==> Waiting for deployments to be ready (this may take 2-3 minutes)..."
kubectl rollout status deployment/vllm-base -n "$NAMESPACE" --timeout=300s &
kubectl rollout status deployment/vllm-lora -n "$NAMESPACE" --timeout=300s &
wait

echo ""
echo "=== Service Endpoints ==="
echo ""
echo "Waiting for LoadBalancer IPs..."
for svc in vllm-base vllm-lora; do
    for i in $(seq 1 60); do
        IP=$(kubectl get svc "$svc" -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
        if [ -n "$IP" ]; then
            echo "  $svc: http://$IP:8000"
            break
        fi
        sleep 5
    done
    if [ -z "$IP" ]; then
        echo "  $svc: LoadBalancer IP pending (check: kubectl get svc $svc -n $NAMESPACE)"
    fi
done

echo ""
echo "Test with:"
echo '  curl -s http://<BASE_IP>:8000/v1/models | python3 -m json.tool'
echo ""
echo "Teardown with:"
echo "  ./scripts/teardown_k8s.sh"
