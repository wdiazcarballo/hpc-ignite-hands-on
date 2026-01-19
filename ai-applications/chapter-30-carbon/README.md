# บทที่ 30: Carbon Footprint และ HPC

Chapter 30: Carbon Verification with HPC

## วัตถุประสงค์การเรียนรู้

1. คำนวณ Carbon Footprint ของ HPC Jobs
2. เข้าใจ Green Computing
3. ใช้ Blockchain สำหรับ Carbon Verification
4. Optimize for Energy Efficiency

## โครงสร้างไฟล์

```
chapter-30-carbon/
├── README.md
├── carbon_calculator.py    # Carbon footprint calculator
├── energy_efficiency.py    # Energy optimization
├── green_scheduling.py     # Green job scheduling
└── sbatch/
    └── carbon_tracked.sbatch
```

## การใช้งาน

```bash
# Calculate carbon footprint
python carbon_calculator.py --job-id 12345

# Optimize for energy
python energy_efficiency.py --gpu-hours 100
```

## LANTA Energy Facts

- Power Usage Effectiveness (PUE): ~1.3
- Cooling: Liquid cooling for GPU nodes
- Location: Thailand (grid carbon intensity varies)
