# บทที่ 23: วัสดุศาสตร์

Chapter 23: Materials Science

## วัตถุประสงค์การเรียนรู้

1. เข้าใจหลักการ DFT Calculations
2. วิเคราะห์ Crystal Structures
3. คำนวณ Electronic Properties
4. ศึกษา Material Databases

## โครงสร้างไฟล์

```
chapter-23-materials-science/
├── README.md
├── crystal_structures.py   # Crystal structure analysis
├── band_structure.py       # Band structure concepts
├── material_properties.py  # Calculate properties
└── sbatch/
    └── dft_job.sbatch
```

## การใช้งาน

```bash
# Create environment
mamba create -n hpc-materials python=3.9 ase pymatgen numpy matplotlib
mamba activate hpc-materials

# Run examples
python crystal_structures.py
python material_properties.py
```

## Key Concepts

- **DFT**: Density Functional Theory
- **Band Gap**: Energy gap between valence and conduction bands
- **Crystal Systems**: Cubic, hexagonal, tetragonal, etc.
