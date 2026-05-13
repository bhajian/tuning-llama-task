# Distributed LoRA Fine-Tuning on Nebius Soperator

Fine-tune Llama 3.1 8B with LoRA across 2 nodes (1x L40s each) over Ethernet, then serve and compare base vs fine-tuned models.

## Prerequisites

- `kubectl` configured to reach the cluster
- HuggingFace token with Llama 3.1 access (`HF_TOKEN`)
- GCP authentication for evaluation (`gcloud auth login`)

---

## Infrastructure Setup

### Step 1: Clone the Nebius solutions repo

This is a fork of [nebius/nebius-solutions-library](https://github.com/nebius/nebius-solutions-library) with Soperator modules for deploying Slurm-on-K8s clusters.

```bash
git clone git@github.com:bhajian/nebius-training.git
cd nebius-training
```

### Step 2: Clone the infrastructure repo

Inside the solutions repo, clone the infrastructure repo that contains all Terraform configs, MLflow, and TensorBoard deployments:

```bash
cd soperator/installations
git clone git@github.com:bhajian/behnam-training-infra.git
cd behnam-training-infra
```

### Step 3: Deploy the Soperator cluster

```bash
export NEBIUS_IAM_TOKEN=$(nebius iam get-access-token)
export TF_VAR_iam_token="$NEBIUS_IAM_TOKEN"
export TF_VAR_vpc_subnet_id=$(nebius vpc subnet list --parent-id project-e00v3cy1pr00enkn7rdbhm --format json | jq -r '.items[0].metadata.id')

terraform init
terraform plan
terraform apply
```

See `behnam-training-infra/README.md` for full cluster configuration details.

### Step 4: Deploy MLflow (experiment tracking)

```bash
cd behnam-training-infra
./mlflow/deploy.sh
```

Access the MLflow UI from your laptop:

```bash
kubectl port-forward svc/mlflow -n mlflow 5000:5000
# Open http://localhost:5000
```

Slurm jobs reach MLflow via internal K8s DNS (`http://mlflow.mlflow.svc.cluster.local:5000`) — no IP configuration needed.

### Step 5: Deploy TensorBoard (profiler visualization)

```bash
./tensorboard/deploy.sh
```

Access TensorBoard from your laptop:

```bash
kubectl port-forward svc/tensorboard -n mlflow 6006:6006
# Open http://localhost:6006 → select PYTORCH_PROFILER from the dropdown
```

### Step 6: Access Grafana (cluster monitoring)

Grafana is deployed automatically by Soperator in the `monitoring-system` namespace. It provides dashboards for cluster health, job timeline, NFS I/O, and node-level CPU/memory/network metrics.

```bash
kubectl port-forward svc/metrics-grafana -n monitoring-system 3000:80
# Open http://localhost:3000
```

Key dashboards:
- **Cluster Health & Overview** — running/finished jobs, job state timeline, per-user/partition breakdown
- **NFS** — disk read/write, NFSd packets, cache hit rates, RPC operations, connections
- **Node Exporter** — CPU, memory, network traffic, disk usage per node

Note: GPU metrics (DCGM) require Soperator worker pods to be running. During inference mode (workers scaled to 0), GPU stats are not available in Grafana — use the Nebius Cloud Console instead.

---

## Important: GPU Ownership

The Soperator worker pods (`worker-0`, `worker-1`) each request `nvidia.com/gpu: 1`. GPUs are **occupied at the Kubernetes level** even when no Slurm job is running. Training and inference cannot share GPUs simultaneously — you must switch between modes:

- **Training mode**: Soperator workers running (GPUs allocated to Slurm)
- **Inference mode**: Soperator workers scaled down (GPUs available for vLLM deployments)

The reconciliation chain is: FluxCD HelmRelease (`nodesets`) → NodeSet CRD → slurm-operator → Kruise StatefulSet → worker pods. To stop workers, you must suspend FluxCD **and** patch the NodeSet CRD. Scaling the StatefulSet alone gets overridden by the slurm-operator.

---

## Dataset Preparation

The Alpaca dataset is split into 90% train / 10% eval to avoid data contamination during evaluation. Run this once from your laptop before training.

```bash
cd training-task
pip install datasets
python dataset/split_dataset.py
```

This produces:
- `dataset/train/alpaca_train.json` — 46,801 examples (used for training)
- `dataset/eval/alpaca_eval.json` — 5,201 examples (used for evaluation)

Then push both files to the shared NFS on the cluster via SCP:

```bash
ssh root@89.169.111.219 "mkdir -p /mnt/data/dataset/train /mnt/data/dataset/eval"
scp dataset/train/alpaca_train.json root@89.169.111.219:/mnt/data/dataset/train/
scp dataset/eval/alpaca_eval.json root@89.169.111.219:/mnt/data/dataset/eval/
```

The JSON files are gitignored — only the split script is committed.

---

## Pre-Training: NCCL Bandwidth Test

Before starting a training run, verify that NCCL communication between nodes is working correctly. This is especially important on Ethernet clusters (no InfiniBand) where misconfigured networking can silently degrade training throughput.

The test runs `all_reduce_perf_mpi` — the same NCCL collective that DDP uses for gradient synchronization — across both nodes and reports bandwidth at various message sizes.

### Run the test

From the login node:

```bash
sbatch ~/training-task/scripts/nccl_test.sbatch
```

Monitor the output:

```bash
tail -f /mnt/data/logs/nccl_test_<jobid>.out
```

### What to look for

The key column in the output is **busbw (GB/s)** — this is the usable bus bandwidth at each message size.

| Message Size | Expected busbw (GB/s) | Notes |
|---|---|---|
| 8 B – 1 KB | ~0.00 | Latency-dominated (~220 μs round-trip) |
| 1 MB | ~0.46 | Small tensors |
| 8 MB | ~0.82 | Approaching link saturation |
| **32 – 64 MB** | **~0.83** | **LoRA gradient range (~50 MB per all-reduce)** |
| 128 – 512 MB | ~0.85 | Peak bandwidth |

**Pass criteria:**
- Zero errors (`#wrong = 0` for all rows)
- Bus bandwidth reaches **0.8+ GB/s** at 32 MB and above
- Bandwidth plateaus and stays flat (no drops at large sizes)

**If the test fails or shows low bandwidth:**
- Check that `NCCL_SOCKET_IFNAME=eth0` matches the actual network interface (`ip addr` on workers)
- Verify `NCCL_IB_DISABLE=1` is set (prevents NCCL from searching for InfiniBand)
- Check for firewall rules or network policies blocking inter-node traffic
- Ensure both workers are on the same subnet

### Expected results on this cluster

With 2x L40S over Ethernet (TCP/Socket transport, 2 NCCL channels):
- **Peak bus bandwidth**: ~0.85 GB/s (~6.8 Gbps)
- **LoRA gradient sync time**: ~60 ms at 50 MB (negligible vs seconds-long forward/backward pass)
- **Average bus bandwidth**: ~0.32 GB/s (across all message sizes including latency-dominated small messages)

---

## Training

### Configuration

All training hyperparameters are defined in YAML config files under `train/configs/`. The default config is `train/configs/default.yaml`:

```yaml
model_id: /mnt/data/Llama-3.1-8B
output_dir: /mnt/data/llama-3.1-8b-lora-adapter
train_data: /mnt/data/dataset/train/alpaca_train.json

max_seq_len: 2048
batch_size: 2
epochs: 5
lr: 2e-4
warmup_ratio: 0.03
max_grad_norm: 1.0
weight_decay: 0.01

lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_targets: [q_proj, k_proj, v_proj, o_proj]

save_every_epoch: true
mlflow_experiment: llama-3.1-8b-lora

profiler_enabled: false
profiler_start_step: 10
profiler_end_step: 20
```

To run an experiment with different hyperparameters, copy the default config and modify it:

```bash
cp train/configs/default.yaml train/configs/experiment_v2.yaml
# Edit experiment_v2.yaml with your changes
sbatch ~/training-task/train/train.sbatch ~/training-task/train/configs/experiment_v2.yaml
```

You can also override individual parameters via CLI without creating a new config file:

```bash
# Inside train.sbatch or directly with torchrun:
finetune.py --config configs/default.yaml --lr 1e-4 --epochs 3 --lora_r 32
```

### Checkpointing

When `save_every_epoch: true` (default), the training script saves:
- `output_dir/checkpoint-epoch-N/` — adapter after each epoch
- `output_dir/best-adapter/` — adapter with the lowest average loss
- `output_dir/` — final adapter after all epochs complete

If a run crashes mid-training, you still have the last completed epoch's checkpoint.

### MLflow Experiment Tracking

Training metrics (loss, learning rate) are automatically logged to MLflow. The tracking URI is set in `train.sbatch` via internal K8s DNS — no manual configuration needed as long as MLflow is deployed (see Infrastructure Setup above).

MLflow logs:
- All hyperparameters at the start of the run
- `train_loss` and `lr` every 10 steps
- `epoch_avg_loss` at the end of each epoch
- Final adapter as an artifact

View experiments from your laptop:

```bash
kubectl port-forward svc/mlflow -n mlflow 5000:5000
# Open http://localhost:5000
```

If MLflow is not deployed, training proceeds normally — the tracking URI will fail silently and training continues without logging.

### Torch Profiler

To profile GPU performance during training, use the profiling config `train/configs/profile_run.yaml`. The profiler captures a window of training steps (default: steps 10-20) and writes TensorBoard-compatible traces to `output_dir/profiler/`.

Profiling is best run as a short test (1 epoch) before your full training run to identify bottlenecks:

1. Run the profiling job:
   ```bash
   sbatch ~/training-task/train/train.sbatch ~/training-task/train/configs/profile_run.yaml
   ```

2. After the job completes, restart TensorBoard to pick up new traces:
   ```bash
   kubectl rollout restart deployment/tensorboard -n mlflow
   ```

3. View profiler results from your laptop:
   ```bash
   kubectl port-forward svc/tensorboard -n mlflow 6006:6006
   ```

4. Open `http://localhost:6006` and select **PYTORCH_PROFILER** from the dropdown to see:
   - GPU utilization and time breakdown
   - Operator-level CPU/GPU timing
   - CUDA kernel view
   - Chrome-trace-style timeline
   - Memory allocation timeline

5. Once satisfied with the profile, run the full training with the default config (`profiler_enabled: false`).

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
ssh root@89.169.111.219 sinfo
# or: kubectl exec -n soperator login-0 -- sinfo
```

Expected:

```
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
main*        up   infinite      2   idle worker-[0-1]
```

### Option A: Traditional (SSH to Login Node)

#### 1. Access the login node

Via SSH (using the Soperator login LoadBalancer):

```bash
ssh root@89.169.111.219
```

Or via kubectl:

```bash
kubectl exec -it -n soperator login-0 -- bash
```

#### 2. Clone the repo and set up the environment

```bash
git clone git@github.com:bhajian/tuning-llama-task.git ~/training-task
export HF_TOKEN=hf_your_token_here
bash ~/training-task/scripts/setup_env.sh
```

`setup_env.sh` is idempotent — it creates a Python venv at `/mnt/data/venv`, installs PyTorch + training dependencies (including `mlflow`, `pyyaml`, `tensorboard`), downloads the Llama 3.1 8B model to `/mnt/data/Llama-3.1-8B`, and verifies the dataset splits are present on NFS. Safe to re-run.

#### 3. Verify GPUs

```bash
srun -N2 --gpus-per-node=1 bash ~/training-task/scripts/check_gpus.sh
```

#### 4. Run NCCL bandwidth test

Validate inter-node NCCL communication before training (see [Pre-Training: NCCL Bandwidth Test](#pre-training-nccl-bandwidth-test) for details):

```bash
sbatch ~/training-task/scripts/nccl_test.sbatch
tail -f /mnt/data/logs/nccl_test_<jobid>.out
```

Confirm bus bandwidth reaches 0.8+ GB/s at 32 MB+ message sizes and zero errors.

#### 5. Submit training job

```bash
# With default config
sbatch ~/training-task/train/train.sbatch

# With a custom config
sbatch ~/training-task/train/train.sbatch ~/training-task/train/configs/experiment_v2.yaml

# Profiling run (1 epoch, profiler enabled)
sbatch ~/training-task/train/train.sbatch ~/training-task/train/configs/profile_run.yaml
```

#### 6. Monitor

```bash
squeue
tail -f /mnt/data/logs/train_<jobid>.out
tail -f /mnt/data/logs/train_<jobid>.err
```

#### 7. Cancel (if needed)

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

The LoRA adapter saves to `/mnt/data/llama-3.1-8b-lora-adapter/` (~52 MB):

```
/mnt/data/llama-3.1-8b-lora-adapter/
  adapter_config.json          # Final adapter
  adapter_model.safetensors
  config.yaml                  # Resolved config (for reproducibility)
  checkpoint-epoch-1/          # Per-epoch checkpoints (if save_every_epoch: true)
  checkpoint-epoch-2/
  ...
  best-adapter/                # Best checkpoint by avg loss
  profiler/                    # Torch profiler traces (if profiler_enabled: true)
```

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

## Evaluation (Claude Sonnet as Judge via Vertex AI)

Evaluates both models on 1000 samples from the held-out eval split (10% of Alpaca, never seen during training) using Claude Sonnet as an impartial judge via GCP Vertex AI. Run this while the inference servers are up.

### Jupyter Notebook (recommended)

```bash
pip install 'anthropic[vertex]' openai datasets matplotlib seaborn pandas jupyter
jupyter notebook inference/evaluation.ipynb
```

Requires GCP authentication (`gcloud auth login`) and env vars:
- `ANTHROPIC_VERTEX_PROJECT_ID` — GCP project ID
- `ANTHROPIC_VERTEX_REGION` — GCP region (default: `us-east5`)

The notebook walks through:
1. Load and sample 1000 examples from the eval split
2. Smoke test: stream one sample from both models to verify LoRA improvement
3. Query both models via their K8s API endpoints
4. Claude Sonnet judges each response (binary YES/NO) with rate limiting and retries
5. Save all results to `results/` (checkpointed — resumable if interrupted)
6. Accuracy bar chart + per-sample agreement breakdown
7. Confusion matrix (base vs LoRA correctness)
8. Throughput analysis (tokens/sec, latency distributions, p95)

### CLI alternative

```bash
python inference/evaluate.py \
    --base-url http://$BASE_IP:8000 \
    --lora-url http://$LORA_IP:8000 \
    --gcp-project $ANTHROPIC_VERTEX_PROJECT_ID \
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
  ├── Sequence packing (2048 ctx, no padding waste)
  ├── YAML config + CLI overrides (reproducible experiments)
  ├── Per-epoch checkpoints + best-adapter tracking
  ├── MLflow experiment tracking (optional, via K8s)
  └── Torch profiler with TensorBoard visualization (optional)

Inference (Kubernetes mode — workers scaled down):
  vLLM in dedicated "inference" namespace
  ├── Deployment A: base Llama 3.1 8B (LoadBalancer)
  └── Deployment B: base + LoRA adapter (LoadBalancer)

Observability (Kubernetes, "mlflow" namespace):
  ├── MLflow tracking server (PVC-backed, port 5000)
  └── TensorBoard (profiler visualization, port 6006)

Evaluation:
  Claude Sonnet judge (Vertex AI) on 1000 Alpaca samples
  └── Binary accuracy scoring (base vs fine-tuned)
```

## Repos

| Repo | Description |
|------|-------------|
| [nebius-training](https://github.com/bhajian/nebius-training) | Fork of nebius-solutions-library with Soperator modules |
| [behnam-training-infra](https://github.com/bhajian/behnam-training-infra) | Terraform configs, MLflow & TensorBoard K8s deployments |
| [tuning-llama-task](https://github.com/bhajian/tuning-llama-task) | Training code, inference configs, evaluation notebook |
