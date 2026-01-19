#!/bin/bash
#SBATCH --job-name=wrf-chem-air-quality
#SBATCH --partition=compute
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=128
#SBATCH --time=04:00:00
#SBATCH --output=wrf_chem_%j.out
#SBATCH --error=wrf_chem_%j.err

# ============================================================
# WRF-Chem Air Quality Modeling - Northern Thailand
# Chapter 0: HPC 101 - กรณีศึกษาการวิเคราะห์คุณภาพอากาศภาคเหนือ
#
# หมายเหตุ: สคริปต์นี้ต้องการ:
#   - WRF-Chem module บน LANTA
#   - ข้อมูล input (boundary conditions, emissions data)
#   - พื้นที่จัดเก็บบน $SCRATCH
#
# สำหรับการเรียนรู้ ให้ศึกษาโครงสร้างสคริปต์ก่อน
# อย่ารันจริงจนกว่าจะมีข้อมูลครบถ้วน
# ============================================================

echo "=========================================="
echo "WRF-Chem Air Quality Simulation"
echo "Region: Northern Thailand"
echo "Start time: $(date)"
echo "=========================================="

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Tasks: $SLURM_NTASKS"

# Set up working directory
WORK_DIR=$SCRATCH/wrf_chem/$SLURM_JOB_ID
mkdir -p $WORK_DIR
cd $WORK_DIR

echo "Working directory: $WORK_DIR"

# Load required modules
module purge
module load WRF-Chem/4.4
module load netCDF/4.8
module load HDF5/1.12.2

echo "Modules loaded:"
module list

# Copy input files to scratch (faster I/O)
echo "Copying input files to scratch..."
# cp -r $HOME/wrf_data/namelist.input .
# cp -r $HOME/wrf_data/wrfinput_d01 .
# cp -r $HOME/wrf_data/wrfbdy_d01 .

# Run WRF-Chem
echo "Starting WRF-Chem simulation..."
echo "Forecast period: 72 hours"

# The actual WRF-Chem execution
# srun wrf.exe

# Post-processing
echo "Running post-processing..."
# python $HOME/scripts/process_output.py --region northern_thailand

# Copy results back to home
echo "Copying results back to home..."
# cp -r wrfout_* $HOME/wrf_results/

# Clean up
echo "Cleaning up scratch..."
# rm -rf $WORK_DIR

echo "=========================================="
echo "Simulation completed at: $(date)"
echo "=========================================="

# Summary
echo ""
echo "Expected outputs:"
echo "  - PM2.5 forecast for next 72 hours"
echo "  - Air quality maps for Northern Thailand"
echo "  - Warning alerts if AQI > 100"
