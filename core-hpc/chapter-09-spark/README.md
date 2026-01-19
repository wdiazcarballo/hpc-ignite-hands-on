# บทที่ 9: Apache Spark

Chapter 9: Apache Spark for Distributed Computing

## วัตถุประสงค์การเรียนรู้

1. เข้าใจ Spark Architecture
2. ใช้ RDD และ Transformations
3. ประยุกต์ Spark DataFrame API
4. รัน Spark บน SLURM Cluster

## โครงสร้างไฟล์

```
chapter-09-spark/
├── README.md
├── spark_basics.py          # PySpark fundamentals
├── spark_dataframe.py       # DataFrame operations
├── spark_wordcount.py       # Classic word count
├── spark_ml_pipeline.py     # Machine learning pipeline
└── sbatch/
    └── spark_cluster.sbatch
```

## การใช้งาน

```bash
# On LANTA
module load Spark/3.3.0

# Run locally
python spark_basics.py

# Submit to SLURM
sbatch sbatch/spark_cluster.sbatch
```

## Spark Architecture

```
┌─────────────────────────────────────────────────┐
│                 Driver Program                  │
│              (SparkContext/Session)             │
└───────────────────────┬─────────────────────────┘
                        │
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
    ┌─────────┐   ┌─────────┐   ┌─────────┐
    │ Worker  │   │ Worker  │   │ Worker  │
    │  Node   │   │  Node   │   │  Node   │
    │ (Tasks) │   │ (Tasks) │   │ (Tasks) │
    └─────────┘   └─────────┘   └─────────┘
```

## Key Concepts

### Lazy Evaluation
```python
# These operations are lazy (not computed yet)
rdd = sc.textFile("data.txt")
filtered = rdd.filter(lambda x: "error" in x)
counts = filtered.map(lambda x: (x, 1))

# Action triggers computation
result = counts.collect()
```
