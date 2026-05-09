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

source /mnt/data/venv/bin/activate

pip install -q vllm 2>/dev/null

python -m vllm.entrypoints.openai.api_server \
    --model /mnt/data/Llama-3.1-8B \
    --dtype bfloat16 \
    --host 0.0.0.0 \
    --port 8001 \
    --max-model-len 2048 \
    --enable-lora \
    --lora-modules lora-adapter=$ADAPTER_DIR
