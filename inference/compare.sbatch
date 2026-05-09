#!/bin/bash
#SBATCH --job-name=model-compare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=/mnt/data/logs/compare_%j.out
#SBATCH --error=/mnt/data/logs/compare_%j.err

set -euo pipefail

BASE_HOST=${BASE_HOST:-"localhost"}
LORA_HOST=${LORA_HOST:-"localhost"}

mkdir -p /mnt/data/logs

source /mnt/data/venv/bin/activate

python /root/training-task/inference/compare.py \
    --base-host "$BASE_HOST" --base-port 8000 \
    --lora-host "$LORA_HOST" --lora-port 8001
