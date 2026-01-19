#!/bin/bash
# Base module loads for HPC Ignite
# Usage: source slurm/module-loads/base.sh

module purge
module load cray-python/3.10.10
module load Miniconda3

echo "Base modules loaded:"
module list
