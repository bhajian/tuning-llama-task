#!/bin/bash
# Download model weights and dataset to shared storage.
# Run this once from a login node before submitting training jobs.
#
# Usage: HF_TOKEN=hf_xxx ./setup_data.sh

set -euo pipefail

MODEL_ID="meta-llama/Llama-3.1-8B"
DATA_DIR="/mnt/data"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "Error: HF_TOKEN is not set. Export your HuggingFace token first."
    echo "  export HF_TOKEN=hf_..."
    exit 1
fi

pip install -q --break-system-packages huggingface_hub datasets transformers sentencepiece protobuf

echo "==> Downloading model: $MODEL_ID"
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    '$MODEL_ID',
    local_dir='$DATA_DIR/Llama-3.1-8B',
    token='$(printenv HF_TOKEN)',
)
print('Model downloaded.')
"

echo "==> Pre-caching Alpaca dataset"
python3 -c "
from datasets import load_dataset
ds = load_dataset('tatsu-lab/alpaca', split='train')
print(f'Dataset cached: {len(ds)} examples')
"

echo "==> Done. Model at $DATA_DIR/Llama-3.1-8B"
