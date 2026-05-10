#!/bin/bash
# Suspend FluxCD, scale down Soperator workers, deploy vLLM serving.
# Run from your laptop.
#
# Usage: ./scripts/serve_k8s.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SOPERATOR_NS="soperator"
INFERENCE_NS="inference"

echo "==> Suspending FluxCD reconciliation"
kubectl patch helmrelease flux-system-soperator-fluxcd-slurm-cluster -n flux-system \
    --type merge -p '{"spec":{"suspend":true}}'
kubectl patch helmrelease flux-system-soperator-fluxcd-nodesets -n flux-system \
    --type merge -p '{"spec":{"suspend":true}}'

echo "==> Scaling NodeSet to 0 (releases GPUs from Soperator workers)"
kubectl patch nodeset worker -n "$SOPERATOR_NS" \
    --type merge -p '{"spec":{"replicas":0}}'

echo "==> Waiting for worker pods to terminate..."
while kubectl get pod worker-0 -n "$SOPERATOR_NS" &>/dev/null; do sleep 2; done
echo "    Workers terminated."

echo "==> Verifying GPUs are available"
GPU_COUNT=$(kubectl get nodes -o jsonpath='{range .items[*]}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}' | awk '{s+=$1} END{print s}')
echo "    Available GPUs: $GPU_COUNT"

echo "==> Creating inference namespace"
kubectl apply -f "$PROJECT_DIR/inference/k8s/namespace.yaml"

echo "==> Deploying vLLM base model server"
kubectl apply -f "$PROJECT_DIR/inference/k8s/base-model.yaml"

echo "==> Deploying vLLM LoRA model server"
kubectl apply -f "$PROJECT_DIR/inference/k8s/lora-model.yaml"

echo "==> Waiting for deployments to be ready (this may take 2-3 minutes)..."
kubectl rollout status deployment/vllm-base -n "$INFERENCE_NS" --timeout=300s &
kubectl rollout status deployment/vllm-lora -n "$INFERENCE_NS" --timeout=300s &
wait

echo ""
echo "=== Service Endpoints ==="
echo ""
echo "Waiting for LoadBalancer IPs..."
for svc in vllm-base vllm-lora; do
    IP=""
    for i in $(seq 1 60); do
        IP=$(kubectl get svc "$svc" -n "$INFERENCE_NS" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
        if [ -n "$IP" ]; then
            echo "  $svc: http://$IP:8000"
            break
        fi
        sleep 5
    done
    if [ -z "$IP" ]; then
        echo "  $svc: LoadBalancer IP pending (check: kubectl get svc $svc -n $INFERENCE_NS)"
    fi
done

echo ""
echo "Test with:"
echo '  curl -s http://<BASE_IP>:8000/v1/models | python3 -m json.tool'
echo ""
echo "Teardown with:"
echo "  ./scripts/teardown_k8s.sh"
