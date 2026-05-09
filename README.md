# Distributed LoRA Fine-Tuning on Nebius Soperator

Fine-tune Llama 3.1 8B with LoRA across 2 nodes (1x L40s each) over Ethernet, then serve and compare base vs fine-tuned models.

## Prerequisites

- Soperator cluster running with 2 worker nodes (L40s GPUs)
- HuggingFace token with Llama 3.1 access
- Shared storage mounted at `/mnt/data`

## Setup

SSH into a login node, clone this repo to your home directory, then download the model:

```bash
export HF_TOKEN=hf_your_token_here
bash ~/training-task/scripts/setup_data.sh
```

Verify GPUs are healthy:

```bash
srun -N2 --gpus-per-node=1 bash ~/training-task/scripts/check_gpus.sh
```

## Training

Submit the 2-node LoRA fine-tuning job:

```bash
sbatch ~/training-task/train/train.sbatch
```

Monitor progress:

```bash
tail -f /mnt/data/logs/train_<jobid>.out
```

The adapter saves to `/mnt/data/llama-3.1-8b-lora-adapter/` when training completes.

## Inference

Start both vLLM servers (submit from login node):

```bash
sbatch ~/training-task/inference/serve_base.sbatch   # port 8000
sbatch ~/training-task/inference/serve_lora.sbatch    # port 8001
```

Wait for both servers to be ready (check logs), then get the hostnames:

```bash
squeue  # note the node names for each serving job
```

Run the comparison:

```bash
BASE_HOST=worker-0 LORA_HOST=worker-1 sbatch ~/training-task/inference/compare.sbatch
```

Or run compare.py directly from a node that can reach both servers:

```bash
python ~/training-task/inference/compare.py --base-host worker-0 --lora-host worker-1
```

## Architecture

```
2 Nodes (1x L40s each, Ethernet)
├── Training: torchrun DDP, NCCL over TCP
│   ├── LoRA (r=16, alpha=32) on q/k/v/o projections
│   ├── BF16 + FlashAttention-2
│   ├── Sequence packing (2048 ctx, no padding waste)
│   └── torch.compile for fused ops
└── Inference: vLLM OpenAI-compatible API
    ├── Node A: base Llama 3.1 8B
    └── Node B: base + LoRA adapter
```
# tuning-llama-task
