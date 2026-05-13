#!/bin/bash
#SBATCH --job-name=nccl-bandwidth-test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/mnt/data/logs/nccl_test_%j.out
#SBATCH --error=/mnt/data/logs/nccl_test_%j.err

set -euo pipefail

# Match training NCCL config exactly (see train.sbatch)
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

mkdir -p /mnt/data/logs

echo "============================================"
echo "  NCCL Bandwidth Test — Pre-Training Check"
echo "============================================"
echo "Date:              $(date)"
echo "Job ID:            $SLURM_JOB_ID"
echo "Nodes:             $SLURM_JOB_NODELIST"
echo "GPUs per node:     $SLURM_GPUS_ON_NODE"
echo "NCCL_IB_DISABLE:   $NCCL_IB_DISABLE"
echo "NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"
echo "============================================"
echo ""

# all_reduce is the collective DDP uses for gradient synchronization.
# Message sizes 8B → 512M cover the full range:
#   - Small messages (8B–1K): dominated by latency, shows round-trip overhead
#   - Medium messages (1M–64M): LoRA gradient range (~50 MB per all-reduce)
#   - Large messages (128M–512M): saturates link, reveals peak bandwidth
#
# -b 8       start at 8 bytes
# -e 512M    end at 512 MB
# -f 2       double message size each step
# -g 1       1 GPU per MPI rank
# -n 20      20 iterations per message size
# -w 5       5 warmup iterations (excluded from timing)

echo ">>> all_reduce_perf_mpi (2 nodes × 1 GPU, TCP/Ethernet)"
echo ""

srun --mpi=pmix \
    all_reduce_perf_mpi -b 8 -e 512M -f 2 -g 1 -n 20 -w 5

echo ""
echo "============================================"
echo "  Key column: busbw (GB/s) = usable bandwidth"
echo "  LoRA gradients are ~50 MB — check the 32M–64M rows"
echo "============================================"
