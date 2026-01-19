#!/bin/bash
# TensorFlow module loads for GPU training
# Usage: source slurm/module-loads/tensorflow.sh

module purge
module load TensorFlow/2.11.0-CUDA-11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0

echo "TensorFlow GPU modules loaded:"
module list

# Verify GPU
python -c "import tensorflow as tf; print(f'GPUs: {len(tf.config.list_physical_devices(\"GPU\"))}')"
