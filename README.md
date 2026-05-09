# Distributed LoRA Fine-Tuning on Nebius Soperator

Fine-tune Llama 3.1 8B with LoRA across 2 nodes (1x L40s each) over Ethernet, then serve and compare base vs fine-tuned models.

## Prerequisites

- Soperator cluster running with 2 worker nodes (L40s GPUs)
- HuggingFace token with Llama 3.1 access (`HF_TOKEN`)
- Shared storage mounted at `/mnt/data`
- `kubectl` configured to reach the cluster (for laptop workflow)

---

## Workflow 1: Traditional (SSH to Login Node)

### 1. Access the login node

```bash
kubectl exec -it -n soperator login-0 -- bash
```

### 2. Clone the repo and set up the environment

```bash
git clone git@github.com:bhajian/tuning-llama-task.git ~/training-task
export HF_TOKEN=hf_your_token_here
bash ~/training-task/scripts/setup_env.sh
```

`setup_env.sh` is idempotent — it creates a Python venv at `/mnt/data/venv`, installs PyTorch + training dependencies, downloads the Llama 3.1 8B model to `/mnt/data/Llama-3.1-8B`, and caches the Alpaca dataset. Safe to re-run.

### 3. Verify GPUs

```bash
srun -N2 --gpus-per-node=1 bash ~/training-task/scripts/check_gpus.sh
```

### 4. Submit training job

```bash
sbatch ~/training-task/train/train.sbatch
```

### 5. Monitor

```bash
squeue
tail -f /mnt/data/logs/train_<jobid>.out
```

### 6. Cancel (if needed)

```bash
scancel <jobid>
```

---

## Workflow 2: From Laptop (kubectl)

No SSH required. All commands run from your laptop.

### 1. Deploy and train

```bash
HF_TOKEN=hf_your_token_here ./scripts/deploy.sh
```

This syncs the code to the login pod, runs `setup_env.sh`, and submits the training job.

### 2. Monitor

```bash
./scripts/status.sh              # queue + last 20 log lines
./scripts/status.sh --logs 100   # queue + last 100 log lines
```

### 3. Cancel

```bash
./scripts/cancel.sh              # cancel all jobs
./scripts/cancel.sh <jobid>      # cancel specific job
```

---

## Inference (after training completes)

The adapter saves to `/mnt/data/llama-3.1-8b-lora-adapter/`.

### Start vLLM servers

```bash
sbatch ~/training-task/inference/serve_base.sbatch   # port 8000
sbatch ~/training-task/inference/serve_lora.sbatch    # port 8001
```

Wait for both servers to be ready (check logs), then note the node names:

```bash
squeue
```

### Run comparison

```bash
BASE_HOST=worker-0 LORA_HOST=worker-1 sbatch ~/training-task/inference/compare.sbatch
```

Or directly:

```bash
python ~/training-task/inference/compare.py --base-host worker-0 --lora-host worker-1
```

---

## Architecture

```
2 Nodes (1x L40s each, Ethernet)
├── Training: torchrun DDP, NCCL over TCP
│   ├── LoRA (r=16, alpha=32) on q/k/v/o projections
│   ├── BF16 + SDPA attention
│   └── Sequence packing (2048 ctx, no padding waste)
└── Inference: vLLM OpenAI-compatible API
    ├── Node A: base Llama 3.1 8B
    └── Node B: base + LoRA adapter
```
