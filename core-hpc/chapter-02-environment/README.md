# บทที่ 2: สภาพแวดล้อม HPC และระบบ LANTA

Chapter 2: HPC Environment and LANTA System

## วัตถุประสงค์การเรียนรู้

1. ใช้ระบบ Module บน LANTA
2. สร้างและจัดการ Conda/Mamba environments
3. ส่งงานด้วย SLURM (sbatch, srun, squeue)
4. จัดการไฟล์บน Lustre filesystem

## โครงสร้างไฟล์

```
chapter-02-environment/
├── README.md
├── check_environment.py    # ตรวจสอบสภาพแวดล้อม
├── slurm_basics.py         # SLURM job information
├── filesystem_demo.py      # File system operations
└── sbatch/
    └── environment_check.sbatch
```

## การใช้งานบน LANTA

```bash
# 1. ตรวจสอบ modules ที่มี
module avail
module spider PyTorch

# 2. โหลด module
module load Miniconda3
module load PyTorch/2.0.1-CUDA-11.7.0

# 3. สร้าง environment
mamba create -n myenv python=3.10
mamba activate myenv

# 4. ส่งงาน
sbatch sbatch/environment_check.sbatch
squeue -u $USER
```

## File Systems

| Path | Usage | Quota | Retention |
|------|-------|-------|-----------|
| `$HOME` | Scripts, configs | 50 GB | Permanent |
| `$SCRATCH` | Data, outputs | 5 TB | 30 days |
| `$PROJECT` | Shared data | Group | Permanent |
