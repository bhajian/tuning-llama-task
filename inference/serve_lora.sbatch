#!/bin/bash
#SBATCH --job-name=vllm-lora
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/mnt/data/logs/serve_lora_%j.out
#SBATCH --error=/mnt/data/logs/serve_lora_%j.err

set -euo pipefail

ADAPTER_DIR="/mnt/data/llama-3.1-8b-lora-adapter"
mkdir -p /mnt/data/logs

srun --container-image=nvcr.io/nvidia/pytorch:24.07-py3 \
     --container-mounts=/mnt/data:/mnt/data,/home:/home \
     bash -c "
        pip install -q vllm && \
        python -m vllm.entrypoints.openai.api_server \
            --model meta-llama/Llama-3.1-8B \
            --dtype bfloat16 \
            --host 0.0.0.0 \
            --port 8001 \
            --max-model-len 2048 \
            --enable-lora \
            --lora-modules lora-adapter=$ADAPTER_DIR
    "
