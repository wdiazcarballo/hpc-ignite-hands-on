#!/bin/bash
#SBATCH --job-name=hello-lanta
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=hello_lanta_%j.out

# ============================================================
# Hello LANTA - ตัวอย่างแรกสำหรับทดสอบการเชื่อมต่อ
# Chapter 0: HPC 101
# ============================================================

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"

# Load Python module
module load Miniconda3

# Run the script
python hello_lanta.py

echo "Job finished at: $(date)"
