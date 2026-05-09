#!/bin/bash
# Quick GPU health check across Slurm nodes.
# Usage: srun -N2 --gpus-per-node=1 ./check_gpus.sh

set -euo pipefail

echo "=== Node: $(hostname) ==="
echo ""
echo "--- nvidia-smi ---"
nvidia-smi
echo ""
echo "--- GPU Memory ---"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,temperature.gpu,power.draw --format=csv,noheader,nounits
echo ""
echo "--- NCCL / Network ---"
echo "NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-not set}"
echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-not set}"
echo "NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-not set}"
echo ""
echo "--- CUDA ---"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]" 2>/dev/null || echo "PyTorch not available in this environment"
