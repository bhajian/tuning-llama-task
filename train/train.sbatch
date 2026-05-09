#!/bin/bash
#SBATCH --job-name=llama-lora-finetune
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/mnt/data/logs/train_%j.out
#SBATCH --error=/mnt/data/logs/train_%j.err

set -euo pipefail

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p /mnt/data/logs

srun bash -c "
    source /mnt/data/venv/bin/activate && \
    torchrun \
        --nnodes=\$SLURM_NNODES \
        --nproc_per_node=1 \
        --node_rank=\$SLURM_NODEID \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        /root/training-task/train/finetune.py
"
