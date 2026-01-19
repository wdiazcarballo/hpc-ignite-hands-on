#!/bin/bash
# PyTorch module loads for GPU training
# Usage: source slurm/module-loads/pytorch.sh

module purge
module load PyTorch/2.0.1-CUDA-11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0
module load NCCL/2.12.12-CUDA-11.7.0

echo "PyTorch GPU modules loaded:"
module list

# Verify GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
