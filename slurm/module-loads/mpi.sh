#!/bin/bash
# MPI module loads for distributed computing
# Usage: source slurm/module-loads/mpi.sh

module purge
module load OpenMPI/4.1.4
module load Miniconda3

echo "MPI modules loaded:"
module list

# Verify MPI
which mpirun
mpirun --version | head -1
