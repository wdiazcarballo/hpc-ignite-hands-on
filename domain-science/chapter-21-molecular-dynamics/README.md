# บทที่ 21: Molecular Dynamics

Chapter 21: Molecular Dynamics Simulations

## วัตถุประสงค์การเรียนรู้

1. เข้าใจหลักการ MD Simulation
2. ใช้ OpenMM สำหรับ MD บน GPU
3. วิเคราะห์ Trajectory Data
4. คำนวณ Properties จาก Simulation

## โครงสร้างไฟล์

```
chapter-21-molecular-dynamics/
├── README.md
├── md_basics.py            # MD fundamentals
├── lennard_jones.py        # Simple LJ simulation
├── water_simulation.py     # Water box simulation
├── trajectory_analysis.py  # Analyze MD trajectories
└── sbatch/
    └── md_gpu.sbatch
```

## การใช้งาน

```bash
# On LANTA
module load CUDA/11.7.0
mamba create -n hpc-md python=3.9 openmm mdtraj numpy matplotlib
mamba activate hpc-md

# Run examples
python md_basics.py
python lennard_jones.py

# GPU simulation
sbatch sbatch/md_gpu.sbatch
```

## MD Simulation Loop

```
1. Initialize positions and velocities
2. Calculate forces: F = -∇U(r)
3. Integrate equations of motion
4. Update positions and velocities
5. Apply constraints/thermostats
6. Repeat for desired time
```
