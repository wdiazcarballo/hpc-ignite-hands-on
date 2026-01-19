# บทที่ 11: Containers สำหรับ HPC

Chapter 11: Containers (Singularity/Apptainer)

## วัตถุประสงค์การเรียนรู้

1. เข้าใจ Container Technology
2. ใช้ Singularity/Apptainer บน HPC
3. Build Custom Containers
4. จัดการ GPU Containers

## โครงสร้างไฟล์

```
chapter-11-containers/
├── README.md
├── singularity_basics.sh   # Basic commands
├── container_demo.py       # Python in container
├── pytorch.def             # PyTorch container definition
├── build_container.sh      # Build script
└── sbatch/
    └── container_gpu.sbatch
```

## การใช้งาน

```bash
# On LANTA
module load Singularity/3.8.3

# Pull container
singularity pull pytorch.sif docker://pytorch/pytorch:latest

# Run container
singularity exec pytorch.sif python script.py

# Run with GPU
singularity exec --nv pytorch.sif python gpu_script.py
```

## Why Containers on HPC?

- **Reproducibility**: Same environment everywhere
- **Portability**: Move between systems
- **Isolation**: No conflicts with system libraries
- **Performance**: Near-native speed
