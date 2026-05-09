#!/bin/bash
# Idempotent environment + data setup. Runs ON the login node.
# Called by deploy.sh via kubectl exec. Can also be run directly.
#
# Usage: HF_TOKEN=hf_xxx bash setup_env.sh

set -euo pipefail

VENV_DIR="/mnt/data/venv"
MODEL_DIR="/mnt/data/Llama-3.1-8B"
MODEL_ID="meta-llama/Llama-3.1-8B"

# --- Python venv ---
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "==> Creating Python venv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "==> Installing PyTorch (CUDA 12.1)"
    pip install -q torch --index-url https://download.pytorch.org/whl/cu121
fi

if ! python -c "import transformers" 2>/dev/null; then
    echo "==> Installing training dependencies"
    pip install -q \
        'transformers==4.44.2' \
        'peft==0.12.0' \
        'accelerate==0.33.0' \
        'huggingface-hub==0.24.0' \
        'tokenizers==0.19.1' \
        'safetensors==0.4.3' \
        datasets sentencepiece protobuf
fi

# --- Model weights ---
if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
    if [ -z "${HF_TOKEN:-}" ]; then
        echo "Error: HF_TOKEN is required for first-time model download."
        echo "  HF_TOKEN=hf_xxx bash setup_env.sh"
        exit 1
    fi
    echo "==> Downloading model: $MODEL_ID"
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_ID', local_dir='$MODEL_DIR', token='$HF_TOKEN')
print('Model downloaded.')
"
else
    echo "==> Model already present at $MODEL_DIR"
fi

# --- Dataset ---
echo "==> Ensuring Alpaca dataset is cached"
python -c "
from datasets import load_dataset
ds = load_dataset('tatsu-lab/alpaca', split='train')
print(f'Dataset ready: {len(ds)} examples')
"

echo "==> Environment ready."
