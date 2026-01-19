# บทที่ 24: AI สำหรับการปกป้องป่า

Chapter 24: AI for Forest Protection

## วัตถุประสงค์การเรียนรู้

1. ใช้ Computer Vision สำหรับ Remote Sensing
2. วิเคราะห์ภาพถ่ายดาวเทียม
3. ตรวจจับการเปลี่ยนแปลงพื้นที่ป่า
4. สร้าง Early Warning System

## โครงสร้างไฟล์

```
chapter-24-ai-forest/
├── README.md
├── satellite_basics.py     # Satellite image processing
├── forest_change.py        # Forest change detection
├── ndvi_analysis.py        # NDVI vegetation index
├── fire_detection.py       # Fire hotspot detection
└── sbatch/
    └── forest_gpu.sbatch
```

## การใช้งาน

```bash
# Create environment
mamba create -n hpc-forest python=3.9 rasterio pytorch torchvision numpy matplotlib
mamba activate hpc-forest

# Run examples
python ndvi_analysis.py
python forest_change.py
```

## Key Vegetation Indices

- **NDVI** = (NIR - Red) / (NIR + Red)
- **EVI** = Enhanced Vegetation Index
- **NDWI** = Normalized Difference Water Index
