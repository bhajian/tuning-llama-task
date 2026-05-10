# Distributed LoRA Fine-Tuning on Nebius Soperator

Fine-tune Llama 3.1 8B with LoRA across 2 nodes (1x L40s each) over Ethernet, then serve and compare base vs fine-tuned models.

## Prerequisites

- Soperator cluster running with 2 worker nodes (L40s GPUs)
- HuggingFace token with Llama 3.1 access (`HF_TOKEN`)
- Shared storage mounted at `/mnt/data`
- `kubectl` configured to reach the cluster

## Important: GPU Ownership

The Soperator worker pods (`worker-0`, `worker-1`) each request `nvidia.com/gpu: 1`. GPUs are **occupied at the Kubernetes level** even when no Slurm job is running. Training and inference cannot share GPUs simultaneously — you must switch between modes:

- **Training mode**: Soperator workers running (GPUs allocated to Slurm)
- **Inference mode**: Soperator workers scaled down (GPUs available for vLLM deployments)

The reconciliation chain is: FluxCD HelmRelease (`nodesets`) → NodeSet CRD → slurm-operator → Kruise StatefulSet → worker pods. To stop workers, you must suspend FluxCD **and** patch the NodeSet CRD. Scaling the StatefulSet alone gets overridden by the slurm-operator.

---

## Training

### Step 1: Ensure Soperator workers are running

If you previously scaled down workers for inference, restore them:

```bash
# Restore NodeSet replicas to 2
kubectl patch nodeset worker -n soperator \
    --type merge -p '{"spec":{"replicas":2}}'

# Resume FluxCD reconciliation
kubectl patch helmrelease flux-system-soperator-fluxcd-nodesets -n flux-system \
    --type merge -p '{"spec":{"suspend":false}}'
kubectl patch helmrelease flux-system-soperator-fluxcd-slurm-cluster -n flux-system \
    --type merge -p '{"spec":{"suspend":false}}'

# Wait for workers to be ready
kubectl wait --for=condition=ready pod/worker-0 pod/worker-1 -n soperator --timeout=300s
```

### Step 2: Verify Slurm is ready

```bash
kubectl exec -n soperator login-0 -- sinfo
```

Expected:

```
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
main*        up   infinite      2   idle worker-[0-1]
```

### Option A: Traditional (SSH to Login Node)

#### 1. Access the login node

```bash
kubectl exec -it -n soperator login-0 -- bash
```

#### 2. Clone the repo and set up the environment

```bash
git clone git@github.com:bhajian/tuning-llama-task.git ~/training-task
export HF_TOKEN=hf_your_token_here
bash ~/training-task/scripts/setup_env.sh
```

`setup_env.sh` is idempotent — it creates a Python venv at `/mnt/data/venv`, installs PyTorch + training dependencies, downloads the Llama 3.1 8B model to `/mnt/data/Llama-3.1-8B`, and caches the Alpaca dataset. Safe to re-run.

#### 3. Verify GPUs

```bash
srun -N2 --gpus-per-node=1 bash ~/training-task/scripts/check_gpus.sh
```

#### 4. Submit training job

```bash
sbatch ~/training-task/train/train.sbatch
```

#### 5. Monitor

```bash
squeue
tail -f /mnt/data/logs/train_<jobid>.out
```

#### 6. Cancel (if needed)

```bash
scancel <jobid>
```

### Option B: From Laptop (kubectl)

No SSH required. All commands run from your laptop.

```bash
# Deploy: syncs code, sets up env, submits job
HF_TOKEN=hf_your_token_here ./scripts/deploy.sh

# Monitor
./scripts/status.sh              # queue + last 20 log lines
./scripts/status.sh --logs 100   # queue + last 100 log lines

# Cancel
./scripts/cancel.sh              # cancel all jobs
./scripts/cancel.sh <jobid>      # cancel specific job
```

### Training output

The LoRA adapter saves to `/mnt/data/llama-3.1-8b-lora-adapter/` (~52 MB).

---

## Inference

Inference runs as Kubernetes Deployments with LoadBalancer services in a dedicated `inference` namespace. This requires freeing GPUs from the Soperator workers.

### Step 1: Suspend FluxCD and scale down Soperator workers

Both FluxCD HelmReleases must be suspended to prevent the slurm-operator from respawning workers:

```bash
# Suspend FluxCD (both slurm-cluster and nodesets)
kubectl patch helmrelease flux-system-soperator-fluxcd-slurm-cluster -n flux-system \
    --type merge -p '{"spec":{"suspend":true}}'
kubectl patch helmrelease flux-system-soperator-fluxcd-nodesets -n flux-system \
    --type merge -p '{"spec":{"suspend":true}}'

# Scale NodeSet to 0 (this is what the slurm-operator watches)
kubectl patch nodeset worker -n soperator \
    --type merge -p '{"spec":{"replicas":0}}'
```

### Step 2: Verify workers are gone and GPUs are free

```bash
# Should return nothing
kubectl get pods -n soperator | grep worker

# Both GPU nodes should show 1 available GPU
kubectl get nodes -o custom-columns='NODE:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu' | grep -v '<none>'
```

Expected:

```
NODE                                 GPU
computeinstance-e00n8wvvx88swma9h1   1
computeinstance-e00x76mf85g7tbcfx9   1
```

Confirm GPUs are not allocated:

```bash
kubectl describe node computeinstance-e00n8wvvx88swma9h1 | grep -A5 'Allocated resources'
kubectl describe node computeinstance-e00x76mf85g7tbcfx9 | grep -A5 'Allocated resources'
```

`nvidia.com/gpu` should show `0` for both Requests and Limits.

### Step 3: NVIDIA smoke test on Kubernetes

Verify GPU access works from a regular Kubernetes pod before deploying vLLM:

```bash
kubectl run gpu-test --rm -it --restart=Never \
    --image=nvidia/cuda:12.1.0-base-ubuntu22.04 \
    --overrides='{
      "spec": {
        "nodeSelector": {"nebius.com/gpu": "true"},
        "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
        "containers": [{
          "name": "gpu-test",
          "image": "nvidia/cuda:12.1.0-base-ubuntu22.04",
          "command": ["nvidia-smi"],
          "resources": {"limits": {"nvidia.com/gpu": "1"}}
        }]
      }
    }' -- nvidia-smi
```

You should see the L40S GPU with 48 GB memory. If this fails, the GPU nodes are not ready for serving.

### Step 4: Deploy vLLM servers

```bash
kubectl apply -f inference/k8s/namespace.yaml
kubectl apply -f inference/k8s/base-model.yaml
kubectl apply -f inference/k8s/lora-model.yaml
```

Or use the script that does steps 1-4 automatically:

```bash
./scripts/serve_k8s.sh
```

### Step 5: Wait for pods to be ready

```bash
kubectl get pods -n inference -w
```

Wait until both show `1/1 Running` (2-3 min for image pull + model load).

### Step 6: Get LoadBalancer IPs

```bash
kubectl get svc -n inference
```

### Step 7: Smoke test the APIs

```bash
BASE_IP=$(kubectl get svc vllm-base -n inference -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
LORA_IP=$(kubectl get svc vllm-lora -n inference -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

# Test base model
curl -s http://$BASE_IP:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/mnt/data/Llama-3.1-8B",
    "prompt": "### Instruction:\nExplain the difference between supervised and unsupervised learning.\n\n### Response:\n",
    "max_tokens": 256,
    "temperature": 0.7
  }' | python3 -m json.tool

# Test LoRA-tuned model
curl -s http://$LORA_IP:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lora-adapter",
    "prompt": "### Instruction:\nExplain the difference between supervised and unsupervised learning.\n\n### Response:\n",
    "max_tokens": 256,
    "temperature": 0.7
  }' | python3 -m json.tool
```

### Teardown: restore Soperator for training

```bash
./scripts/teardown_k8s.sh
```

Or manually:

```bash
# Remove vLLM deployments
kubectl delete -f inference/k8s/lora-model.yaml
kubectl delete -f inference/k8s/base-model.yaml

# Restore NodeSet replicas
kubectl patch nodeset worker -n soperator \
    --type merge -p '{"spec":{"replicas":2}}'

# Resume FluxCD
kubectl patch helmrelease flux-system-soperator-fluxcd-nodesets -n flux-system \
    --type merge -p '{"spec":{"suspend":false}}'
kubectl patch helmrelease flux-system-soperator-fluxcd-slurm-cluster -n flux-system \
    --type merge -p '{"spec":{"suspend":false}}'

# Wait for workers
kubectl wait --for=condition=ready pod/worker-0 pod/worker-1 -n soperator --timeout=300s
```

---

## Evaluation (GPT-4o as Judge)

Evaluates both models on 1000 Alpaca samples using GPT-4o as an impartial judge. Run this while the inference servers are up.

### Jupyter Notebook (recommended)

```bash
pip install openai datasets matplotlib seaborn pandas jupyter
OPENAI_API_KEY=sk-xxx jupyter notebook inference/evaluation.ipynb
```

The notebook walks through:
1. Load and sample 1000 Alpaca examples
2. Query both models via their K8s API endpoints
3. GPT-4o judges each response (binary YES/NO)
4. Save all results to `results/`
5. Accuracy bar chart + per-sample agreement breakdown
6. Confusion matrix (base vs LoRA correctness)
7. Throughput analysis (tokens/sec, latency distributions, p95)

### CLI alternative

```bash
OPENAI_API_KEY=sk-xxx python inference/evaluate.py \
    --base-url http://$BASE_IP:8000 \
    --lora-url http://$LORA_IP:8000 \
    --samples 1000
```

---

## Architecture

```
2 Nodes (1x L40s each, Ethernet)

Training (Soperator/Slurm mode):
  torchrun DDP, NCCL over TCP
  ├── LoRA (r=16, alpha=32) on q/k/v/o projections
  ├── BF16 + SDPA attention
  ├── Gradient checkpointing
  └── Sequence packing (2048 ctx, no padding waste)

Inference (Kubernetes mode — workers scaled down):
  vLLM in dedicated "inference" namespace
  ├── Deployment A: base Llama 3.1 8B (LoadBalancer)
  └── Deployment B: base + LoRA adapter (LoadBalancer)

Evaluation:
  GPT-4o judge on 1000 Alpaca samples
  └── Binary accuracy scoring (base vs fine-tuned)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for full infrastructure details.
