# บทที่ 20: เคมีคอมพิวเตอร์

Chapter 20: Computational Chemistry

## วัตถุประสงค์การเรียนรู้

1. เข้าใจ Molecular Structure และ Energy Calculations
2. ใช้ RDKit สำหรับ Cheminformatics
3. คำนวณ Molecular Properties
4. วิเคราะห์ Drug-like Properties

## โครงสร้างไฟล์

```
chapter-20-computational-chemistry/
├── README.md
├── molecular_basics.py      # Basic molecular structures
├── property_calculation.py  # Calculate molecular properties
├── similarity_search.py     # Molecular similarity
├── drug_analysis.py         # Drug-likeness analysis
└── sbatch/
    └── chem_job.sbatch
```

## การใช้งาน

```bash
# Create environment
mamba create -n hpc-chem python=3.9 rdkit numpy pandas matplotlib
mamba activate hpc-chem

# Run examples
python molecular_basics.py
python property_calculation.py
```

## Dependencies

- RDKit: Cheminformatics
- NumPy: Numerical computing
- Pandas: Data handling
- Matplotlib: Visualization
