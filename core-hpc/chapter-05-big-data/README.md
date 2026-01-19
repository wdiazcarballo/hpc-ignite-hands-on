# บทที่ 5: การประมวลผล Big Data

Chapter 5: Big Data Processing

## วัตถุประสงค์การเรียนรู้

1. เข้าใจ 5V ของ Big Data
2. ใช้ Pandas สำหรับข้อมูลขนาดกลาง
3. ประยุกต์ Chunk Processing สำหรับข้อมูลใหญ่
4. ใช้ Out-of-Core Computing

## โครงสร้างไฟล์

```
chapter-05-big-data/
├── README.md
├── pandas_basics.py         # Pandas fundamentals
├── chunk_processing.py      # Processing large files in chunks
├── memory_efficient.py      # Memory-efficient techniques
├── generate_large_data.py   # Generate test data
└── sbatch/
    └── big_data_job.sbatch
```

## การใช้งาน

```bash
# Create environment
mamba env create -f ../../environments/base.yaml
mamba activate hpc-ignite-base

# Generate sample data
python generate_large_data.py --size 1000000

# Run examples
python pandas_basics.py
python chunk_processing.py

# On SLURM
sbatch sbatch/big_data_job.sbatch
```

## แนวคิดหลัก: 5V ของ Big Data

1. **Volume** - ปริมาณข้อมูล
2. **Velocity** - ความเร็วในการสร้างข้อมูล
3. **Variety** - ความหลากหลายของข้อมูล
4. **Veracity** - ความถูกต้องของข้อมูล
5. **Value** - มูลค่าของข้อมูล
