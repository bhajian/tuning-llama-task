# Architecture: Distributed LoRA Fine-Tuning on Nebius Soperator

## Overview

This project demonstrates distributed LoRA fine-tuning of Llama 3.1 8B across 2 GPU nodes using Nebius Soperator — a Slurm-on-Kubernetes platform that brings HPC-style workload management to cloud-native infrastructure. The system supports training, serving, and automated evaluation with GPT-4o as a judge.

---

## Infrastructure Stack

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Nebius Cloud (eu-north1)                     │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              Managed Kubernetes (v1.32.9)                   │    │
│  │                                                             │    │
│  │   ┌──────────────────────────────────────────────────┐      │    │
│  │   │           Soperator 3.0.3 (Slurm 25.11.3)       │      │    │
│  │   │                                                  │      │    │
│  │   │  ┌────────────┐  ┌────────────┐  ┌──────────┐   │      │    │
│  │   │  │ Controller │  │  Login x2  │  │ Worker x2│   │      │    │
│  │   │  │  (slurmctld)│  │  (sshd)    │  │ (slurmd) │   │      │    │
│  │   │  │  4 vCPU    │  │  30 vCPU   │  │ 15 vCPU  │   │      │    │
│  │   │  │  11 GiB    │  │  112 GiB   │  │ 84 GiB   │   │      │    │
│  │   │  │  No GPU    │  │  No GPU    │  │ 1x L40S  │   │      │    │
│  │   │  └────────────┘  └────────────┘  └──────────┘   │      │    │
│  │   └──────────────────────────────────────────────────┘      │    │
│  │                                                             │    │
│  │   FluxCD (20 HelmReleases) │ DCGM Exporter │ MariaDB       │    │
│  │   Cert-Manager │ NFS Server │ OpenTelemetry │ Kruise        │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Storage Layer                              │   │
│  │  Filestore (Jail):  1 TiB  — OS, packages, shared libraries  │   │
│  │  Filestore (Data):  1 TiB  — model weights, adapters, logs   │   │
│  │  NFS (Home):        930 GiB — user home directories           │   │
│  │  Network SSD:       930 GiB x2 — container image storage     │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Cluster Nodes

The cluster has **11 nodes** across 4 functional roles:

| Role | Count | Instance Type | vCPU | Memory | GPU | Purpose |
|------|-------|--------------|------|--------|-----|---------|
| GPU Workers | 2 | `gpu-l40s-d` | 16 | 96 GiB | 1x L40S | Training and inference compute |
| System | 5 | `cpu-d3` | 8 | 32 GiB | — | Soperator control plane, jail mounts, jobs |
| Login | 2 | `cpu-d3` | 32 | 128 GiB | — | User access, job submission |
| Controller | 1 | `cpu-d3` | 4 | 16 GiB | — | Slurm controller (slurmctld) |
| Accounting | 1 | `cpu-d3` | 8 | 32 GiB | — | MariaDB for Slurm accounting |

### GPU Nodes Detail

| Property | Value |
|----------|-------|
| GPU Model | NVIDIA L40S (Ada Lovelace) |
| VRAM | 48 GB GDDR6 (44.4 GB usable) |
| BF16 Performance | 362 TFLOPS |
| FP32 Performance | 91 TFLOPS |
| Memory Bandwidth | 864 GB/s |
| Interconnect | Ethernet (no InfiniBand) |
| CUDA Driver | 570.211.01 |
| CUDA Toolkit | 12.8 |
| Resource Preset | `1gpu-16vcpu-96gb` |
| Nebius Instance Type | `gpu-l40s-d` |

---

## How Soperator Works: Slurm on Kubernetes

Soperator is Nebius's solution for running HPC Slurm workloads on Kubernetes. It maps traditional Slurm concepts to Kubernetes primitives:

### The Mapping

```
Traditional HPC Cluster          Soperator on Kubernetes
─────────────────────          ─────────────────────────
Bare-metal login node    →     Login Pod (StatefulSet, 2 replicas)
Bare-metal head node     →     Controller Pod (StatefulSet, 1 replica)
Bare-metal compute nodes →     Worker Pods (Kruise StatefulSet, 2 replicas)
Shared NFS filesystem    →     PersistentVolumes (Filestore + NFS)
SLURM daemon (slurmd)    →     Container in worker pod
SLURM controller         →     Container in controller pod
Job scheduler            →     slurmctld inside controller pod
GPU drivers              →     NVIDIA device plugin + host driver mount
```

### Architecture Flow

```
User (laptop)
    │
    │  kubectl exec -n soperator login-0 -- bash
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  Login Pod (login-0)                                             │
│  ┌──────────────────────────────────────────────────────┐        │
│  │  Slurm client tools: sbatch, squeue, scancel, sinfo  │        │
│  │  Shared filesystem access: /mnt/data, /home           │        │
│  └──────────────────────────┬───────────────────────────┘        │
│                              │ sbatch train.sbatch                │
│                              ▼                                    │
│  ┌──────────────────────────────────────────────────────┐        │
│  │  Controller Pod (controller-0)                        │        │
│  │  slurmctld — schedules jobs across worker pods        │        │
│  │  Accounting DB (MariaDB) for job history              │        │
│  └──────────────┬───────────────────────┬───────────────┘        │
│                  │ srun                  │ srun                   │
│                  ▼                       ▼                        │
│  ┌─────────────────────┐  ┌─────────────────────┐                │
│  │  Worker Pod (worker-0) │  │  Worker Pod (worker-1) │           │
│  │  slurmd daemon         │  │  slurmd daemon         │           │
│  │  1x L40S GPU           │  │  1x L40S GPU           │           │
│  │  15 vCPU, 84 GiB RAM  │  │  15 vCPU, 84 GiB RAM  │           │
│  │  /mnt/data (shared)    │  │  /mnt/data (shared)    │           │
│  └─────────────────────┘  └─────────────────────┘                │
│            │                          │                           │
│            └──────── NCCL TCP ────────┘                           │
│                   (gradient sync)                                 │
└──────────────────────────────────────────────────────────────────┘
```

### Key Concepts

**Jail Filesystem**: Soperator uses an overlay filesystem called "jail" that provides a consistent Linux environment across all pods. The jail contains system packages, libraries, and shared tools. It's mounted from a Nebius Filestore (1 TiB) and overlaid on each pod's root filesystem. When a Slurm job runs on a worker, it executes inside this jail — seeing `/mnt/data`, `/home`, and system tools as if on a bare-metal HPC node.

**Kruise StatefulSet**: Workers are managed by OpenKruise's StatefulSet (not standard Kubernetes StatefulSet). Kruise provides in-place pod updates and advanced scheduling features needed for HPC workloads.

**FluxCD GitOps**: The entire Soperator stack is deployed via 20 FluxCD HelmReleases, providing declarative, version-controlled infrastructure. Key components include:
- `slurm-cluster` — the core Soperator chart (controller, login, workers)
- `dcgm-exporter` — NVIDIA GPU metrics collection
- `nfs-server` — NFS provisioner for home directories
- `opentelemetry-collector` — log aggregation (events, jail logs, system logs)
- `kruise` — advanced workload management for worker pods
- `cert-manager`, `security-profiles-operator` — security infrastructure

**Slurm Partitions**: Two partitions are configured:
- `main` (default) — both workers, unlimited time, preemptable (REQUEUE)
- `hidden` — same workers, non-preemptable, used for system checks

The `main` partition exposes: 32 CPUs, 168 GiB memory, 2 GPUs across 2 nodes.

---

## Storage Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Storage Volumes                           │
│                                                                 │
│  ┌─────────────────────────────────────────────────┐            │
│  │  Jail Filestore (1 TiB, ReadWriteMany)          │            │
│  │  Path: /mnt/jail                                 │            │
│  │  Contains: OS overlay, system packages,          │            │
│  │            Python, shared libraries              │            │
│  │  Populated by: soperator-populate-jail job        │            │
│  └─────────────────────────────────────────────────┘            │
│                                                                 │
│  ┌─────────────────────────────────────────────────┐            │
│  │  Data Filestore (1 TiB, ReadWriteMany)          │            │
│  │  Path: /mnt/data (inside jail)                   │            │
│  │  Contains:                                       │            │
│  │    /mnt/data/Llama-3.1-8B/        (16 GB)       │            │
│  │    /mnt/data/llama-3.1-8b-lora-adapter/ (52 MB) │            │
│  │    /mnt/data/venv/                (PyTorch env)  │            │
│  │    /mnt/data/logs/                (job outputs)  │            │
│  └─────────────────────────────────────────────────┘            │
│                                                                 │
│  ┌─────────────────────────────────────────────────┐            │
│  │  NFS Home (930 GiB, ReadWriteMany)              │            │
│  │  Path: /home                                     │            │
│  │  Contains: user home directories, code repos     │            │
│  └─────────────────────────────────────────────────┘            │
│                                                                 │
│  ┌─────────────────────────────────────────────────┐            │
│  │  Image Storage (930 GiB x2, ReadWriteOnce)      │            │
│  │  Per-worker network SSD (io-m3)                  │            │
│  │  Contains: container images (enroot)             │            │
│  └─────────────────────────────────────────────────┘            │
│                                                                 │
│  ┌─────────────────────────────────────────────────┐            │
│  │  Local Data (1 TiB x2, ReadWriteOnce)           │            │
│  │  Per-worker network SSD                          │            │
│  │  Contains: local scratch space                   │            │
│  └─────────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

All pods in the Soperator namespace see the same jail and data filestores. Home directories are served via an in-cluster NFS server backed by a Nebius compute disk. This shared storage model means model weights downloaded once are immediately available to all workers — no per-node data staging.

---

## Training Architecture

### Distributed Data Parallel (DDP) over Ethernet

```
┌─────────────────────────────────────────────────────┐
│                   torchrun launcher                  │
│              (via srun on each worker)               │
└────────────┬────────────────────────┬───────────────┘
             │                        │
     ┌───────▼───────┐       ┌───────▼───────┐
     │   Worker-0     │       │   Worker-1     │
     │   Rank 0       │       │   Rank 1       │
     │                │       │                │
     │  ┌──────────┐  │       │  ┌──────────┐  │
     │  │ Llama 8B │  │       │  │ Llama 8B │  │
     │  │  (bf16)  │  │       │  │  (bf16)  │  │
     │  │ + LoRA   │  │       │  │ + LoRA   │  │
     │  └────┬─────┘  │       │  └────┬─────┘  │
     │       │        │       │       │        │
     │  ┌────▼─────┐  │       │  ┌────▼─────┐  │
     │  │   DDP    │  │       │  │   DDP    │  │
     │  │ wrapper  │◄─┼─NCCL──┼─►│ wrapper  │  │
     │  └────┬─────┘  │  TCP  │  └────┬─────┘  │
     │       │        │       │       │        │
     │  ┌────▼─────┐  │       │  ┌────▼─────┐  │
     │  │  L40S    │  │       │  │  L40S    │  │
     │  │  48 GB   │  │       │  │  48 GB   │  │
     │  └──────────┘  │       │  └──────────┘  │
     └────────────────┘       └────────────────┘
          Data Shard 1              Data Shard 2
```

### Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | Llama 3.1 8B | 8B params, fits in L40S VRAM with LoRA |
| Precision | BF16 | Half-precision, ~16 GB model weights, full L40S tensor core throughput |
| LoRA Rank (r) | 16 | Good quality/efficiency tradeoff for instruction tuning |
| LoRA Alpha | 32 | 2x rank, standard scaling factor |
| LoRA Targets | q, k, v, o projections | All attention projections for maximum adaptation |
| Trainable Params | ~25M (0.3% of 8B) | Enables training on 48 GB GPUs |
| Batch Size | 2 per GPU (4 effective) | Fits in VRAM with gradient checkpointing |
| Sequence Length | 2048 tokens | Full Llama 3.1 context for instruction following |
| Epochs | 3 | Standard for Alpaca-style instruction tuning |
| Learning Rate | 2e-4 | Standard for LoRA fine-tuning |
| Scheduler | Cosine with warmup (3%) | Smooth LR decay |
| Dataset | Alpaca (52K examples) | Instruction-following dataset |
| Attention | SDPA (PyTorch native) | Equivalent to FlashAttention, no extra install |
| Gradient Checkpointing | Enabled | Trades ~30% compute for ~70% memory savings |
| NCCL Transport | TCP over Ethernet | No InfiniBand; `NCCL_IB_DISABLE=1` |

### Memory Budget (per L40S GPU)

```
Component                    Memory
─────────────────────────    ──────
Model weights (bf16)         ~16 GB
LoRA adapters                ~0.1 GB
Activations (checkpointed)   ~5 GB
LoRA gradients + optimizer   ~1 GB
CUDA runtime + buffers       ~1 GB
─────────────────────────    ──────
Total                        ~23 GB / 44.4 GB usable
Headroom                     ~21 GB (47% free)
```

### Optimization Techniques

1. **LoRA (Low-Rank Adaptation)**: Only trains 0.3% of parameters. Optimizer states (AdamW maintains 2 copies per trainable param) consume ~100 MB instead of ~32 GB for full fine-tuning.

2. **Gradient Checkpointing**: Discards intermediate activations during forward pass and recomputes them during backward. Reduces activation memory from ~28 GB to ~5 GB. Costs ~30% extra compute but enables training that would otherwise OOM.

3. **Sequence Packing**: Concatenates all tokenized examples into a continuous stream, then slices into fixed 2048-token chunks. Zero padding waste — every GPU cycle processes real data, not pad tokens.

4. **BF16 Precision**: Uses bfloat16 throughout, halving memory vs fp32. L40S has dedicated bf16 tensor cores running at 362 TFLOPS.

5. **SDPA (Scaled Dot Product Attention)**: PyTorch's native fused attention kernel. Automatically selects the optimal backend (FlashAttention, memory-efficient, or math). No separate package installation required.

6. **Ethernet-Optimized DDP**: LoRA's tiny gradient footprint (~50 MB per sync vs ~16 GB for full model) makes Ethernet viable. NCCL all-reduce over TCP adds negligible overhead because there's so little data to transfer.

### Training Results

- **GPU Utilization**: 100% sustained on both L40S GPUs
- **Memory Utilization**: 60-80% (23 GB / 44.4 GB)
- **Loss Progression**: 1.35 (epoch 1 start) → 1.18 (epoch 3 end)
- **Training Time**: ~75 minutes for 3 epochs (1,548 steps)
- **Adapter Size**: 52 MB (vs 16 GB base model)

---

## Inference Architecture

After training, we transition from Slurm to native Kubernetes for serving:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Kubernetes (inference namespace)              │
│                                                                 │
│  ┌────────────────────────┐    ┌────────────────────────┐       │
│  │  Deployment: vllm-base │    │  Deployment: vllm-lora │       │
│  │  vLLM OpenAI Server    │    │  vLLM OpenAI Server    │       │
│  │                        │    │                        │       │
│  │  Model: Llama 3.1 8B   │    │  Model: Llama 3.1 8B   │       │
│  │  (base weights only)   │    │  + LoRA adapter         │       │
│  │                        │    │                        │       │
│  │  1x L40S GPU           │    │  1x L40S GPU           │       │
│  │  Port 8000             │    │  Port 8000             │       │
│  └───────────┬────────────┘    └───────────┬────────────┘       │
│              │                              │                    │
│  ┌───────────▼────────────┐    ┌───────────▼────────────┐       │
│  │  Service: vllm-base    │    │  Service: vllm-lora    │       │
│  │  Type: LoadBalancer    │    │  Type: LoadBalancer    │       │
│  │  External IP assigned  │    │  External IP assigned  │       │
│  └───────────┬────────────┘    └───────────┬────────────┘       │
└──────────────┼─────────────────────────────┼────────────────────┘
               │                              │
               ▼                              ▼
        http://<IP>:8000/v1           http://<IP>:8000/v1
        (OpenAI-compatible)           (OpenAI-compatible)
```

### Slurm → Kubernetes Transition

The serving workflow requires freeing GPUs from Soperator workers. The reconciliation chain is: FluxCD HelmRelease → NodeSet CRD → slurm-operator → Kruise StatefulSet → worker pods. To release GPUs:

1. **Suspend FluxCD**: Suspend both `slurm-cluster` and `nodesets` HelmReleases to prevent the operator from respawning workers
2. **Scale down NodeSet**: `kubectl patch nodeset worker -n soperator --type merge -p '{"spec":{"replicas":0}}'` — this is what the slurm-operator watches
3. **Deploy vLLM**: Kubernetes Deployments in the `inference` namespace with `nvidia.com/gpu: 1` resource requests claim the freed GPUs
4. **Expose via LoadBalancer**: Each deployment gets an external IP via Nebius LoadBalancer
5. **Restore**: Delete deployments, restore NodeSet replicas to 2, resume both FluxCD HelmReleases

### vLLM Serving Configuration

| Parameter | Base Server | LoRA Server |
|-----------|-------------|-------------|
| Image | `vllm/vllm-openai:v0.20.2` | Same |
| Model | `/mnt/data/Llama-3.1-8B` | `/mnt/data/Llama-3.1-8B` |
| LoRA | — | `/mnt/data/llama-3.1-8b-lora-adapter` |
| Precision | BF16 | BF16 |
| Max Seq Len | 2048 | 2048 |
| API | OpenAI-compatible (`/v1/completions`) | OpenAI-compatible |
| GPU | 1x L40S | 1x L40S |

---

## Evaluation Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Evaluation Pipeline                    │
│                                                          │
│   Alpaca Dataset (1000 random samples, seed=42)          │
│         │                                                │
│         ▼                                                │
│   ┌─────────────────────────────────────────────┐        │
│   │  For each sample:                            │        │
│   │                                              │        │
│   │  ┌───────────┐          ┌───────────┐        │        │
│   │  │ Base Model │          │ LoRA Model │       │        │
│   │  │ (vLLM)     │          │ (vLLM)     │       │        │
│   │  └─────┬─────┘          └─────┬─────┘        │        │
│   │        │                      │               │        │
│   │        ▼                      ▼               │        │
│   │  base_response          lora_response         │        │
│   │        │                      │               │        │
│   │        └──────────┬───────────┘               │        │
│   │                   ▼                           │        │
│   │         ┌──────────────────┐                  │        │
│   │         │    GPT-4o Judge   │                  │        │
│   │         │                  │                  │        │
│   │         │  "Is this response│                  │        │
│   │         │  correct, relevant│                  │        │
│   │         │  and helpful?"   │                  │        │
│   │         │                  │                  │        │
│   │         │  → YES (1) or    │                  │        │
│   │         │    NO  (0)       │                  │        │
│   │         └──────────────────┘                  │        │
│   └─────────────────────────────────────────────┘        │
│                                                          │
│   Results:                                               │
│     Base Accuracy  = correct_base / total × 100          │
│     LoRA Accuracy  = correct_lora / total × 100          │
│     Improvement    = LoRA Accuracy - Base Accuracy       │
│                                                          │
│   Output: eval_results.json (per-sample details)         │
└──────────────────────────────────────────────────────────┘
```

The evaluation uses 8 parallel workers for throughput. Each sample requires 2 vLLM API calls (base + LoRA) and 2 GPT-4o API calls (judge each response). For 1000 samples, this means ~4000 API calls total.

---

## Network Architecture

```
                    Internet
                       │
                       ▼
              ┌────────────────┐
              │ Nebius Cloud    │
              │ Load Balancer   │
              └───┬────────┬───┘
                  │        │
         ┌────────▼──┐  ┌──▼────────┐
         │ vllm-base │  │ vllm-lora │
         │ :8000     │  │ :8000     │
         └───────────┘  └───────────┘

    Within cluster (training):

    worker-0 ◄──── NCCL TCP (eth0) ────► worker-1
    (rank 0)     gradient all-reduce      (rank 1)
                port 29500 (master)
```

### NCCL Configuration (Training)

| Setting | Value | Rationale |
|---------|-------|-----------|
| `NCCL_IB_DISABLE` | `1` | No InfiniBand on L40S nodes |
| `NCCL_SOCKET_IFNAME` | `eth0` | Use Ethernet for NCCL transport |
| `NCCL_DEBUG` | `INFO` | Verbose logging for debugging |
| Backend | TCP | Default when IB disabled |

Ethernet is not a bottleneck for LoRA training because gradient synchronization only transfers ~50 MB per step (25M trainable LoRA params × 2 bytes). For comparison, full-parameter DDP would need to sync ~16 GB per step, which would saturate Ethernet and drop GPU utilization.

---

## Operations

### Two Workflows

```
Workflow 1: Traditional (SSH)              Workflow 2: Laptop (kubectl)
─────────────────────────────              ──────────────────────────────
kubectl exec login-0 -- bash               ./scripts/deploy.sh
git clone <repo>                            (syncs code + submits job)
bash setup_env.sh                          ./scripts/status.sh
sbatch train.sbatch                        ./scripts/cancel.sh
squeue / tail -f logs
scancel <jobid>
```

### Lifecycle

```
1. SETUP        HF_TOKEN=xxx ./scripts/deploy.sh
                (or: login → git clone → setup_env.sh → sbatch)

2. TRAIN        sbatch train.sbatch
                Monitor: tail -f /mnt/data/logs/train_<id>.out
                Output: /mnt/data/llama-3.1-8b-lora-adapter/

3. SERVE        ./scripts/serve_k8s.sh
                (scales down Slurm workers → deploys vLLM on K8s)

4. EVALUATE     OPENAI_API_KEY=xxx python evaluate.py \
                  --base-url http://<IP>:8000 \
                  --lora-url http://<IP>:8000

5. TEARDOWN     ./scripts/teardown_k8s.sh
                (removes vLLM → restores Slurm workers)
```

---

## Software Versions

| Component | Version |
|-----------|---------|
| Kubernetes | 1.32.9 |
| Soperator | 3.0.3 |
| Slurm | 25.11.3 |
| NVIDIA Driver | 570.211.01 |
| CUDA | 12.8 |
| PyTorch | 2.x (cu121 wheel) |
| Transformers | 4.44.2 |
| PEFT | 0.12.0 |
| Accelerate | 0.33.0 |
| vLLM | 0.20.2 |
| FluxCD | HelmRelease v2 |
| OpenKruise | 1.8.0 |
| cert-manager | 1.19.5 |

---

## Why L40S for Llama 3.1 8B

The NVIDIA L40S (48 GB VRAM) is well-suited for this workload:

- **VRAM Capacity**: 8B params in bf16 = ~16 GB. With LoRA + gradient checkpointing, total usage is ~23 GB — comfortably within 48 GB. Full fine-tuning of 8B would need 80+ GB (A100/H100 territory).
- **BF16 Tensor Cores**: 362 TFLOPS bf16 throughput. LLM training is matmul-bound, and L40S delivers strong throughput for the price point.
- **Ada Lovelace Architecture**: Supports SDPA/FlashAttention natively, efficient memory access patterns for transformer workloads.
- **Cost Efficiency**: L40S is significantly cheaper than A100-80GB or H100, while having sufficient VRAM for LoRA fine-tuning and single-GPU inference of 8B models.
- **Inference Ready**: The same GPU used for training can serve the model via vLLM with full bf16 precision — no quantization needed.
