# บทที่ 6: การสร้างภาพข้อมูล

Chapter 6: Data Visualization

## วัตถุประสงค์การเรียนรู้

1. เข้าใจหลักการ Data-Ink Ratio
2. ใช้ Matplotlib สร้างกราฟพื้นฐาน
3. สร้าง Interactive Visualization
4. เลือกประเภทกราฟที่เหมาะสม

## โครงสร้างไฟล์

```
chapter-06-visualization/
├── README.md
├── matplotlib_basics.py     # Basic plotting
├── chart_types.py           # Different chart types
├── hpc_dashboard.py         # HPC monitoring visualization
├── publication_quality.py   # Publication-ready figures
└── sbatch/
    └── viz_job.sbatch
```

## การใช้งาน

```bash
# Create environment
mamba env create -f ../../environments/base.yaml
mamba activate hpc-ignite-base

# Run examples
python matplotlib_basics.py
python chart_types.py

# Generate figures (saves to PNG)
python hpc_dashboard.py --output dashboard.png
```

## หลักการ Data-Ink Ratio

> "Above all else, show the data" - Edward Tufte

Data-Ink Ratio = (ink used for data) / (total ink)

เป้าหมาย: ลด "chartjunk" ให้เหลือน้อยที่สุด
