# บทที่ 26: เกษตรอัจฉริยะ

Chapter 26: Smart Agriculture

## วัตถุประสงค์การเรียนรู้

1. ใช้ IoT Data สำหรับการเกษตร
2. วิเคราะห์ข้อมูล Crop Yield
3. สร้าง Prediction Models
4. ประยุกต์ Remote Sensing

## โครงสร้างไฟล์

```
chapter-26-smart-agriculture/
├── README.md
├── crop_analysis.py        # Crop yield analysis
├── weather_impact.py       # Weather impact on crops
├── yield_prediction.py     # ML yield prediction
├── irrigation_scheduler.py # Smart irrigation
└── sbatch/
    └── agri_job.sbatch
```

## การใช้งาน

```bash
# Create environment
mamba create -n hpc-agri python=3.9 scikit-learn pandas numpy matplotlib
mamba activate hpc-agri

# Run examples
python crop_analysis.py
python yield_prediction.py
```

## Northern Thailand Crops

- **ข้าว (Rice)**: Main crop, rainy season
- **ลำไย (Longan)**: Major fruit export
- **ลิ้นจี่ (Lychee)**: Premium fruit
- **กาแฟ (Coffee)**: Highland crop
