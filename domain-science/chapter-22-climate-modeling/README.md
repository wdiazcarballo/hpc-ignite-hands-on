# บทที่ 22: การจำลองภูมิอากาศ

Chapter 22: Climate Modeling

## วัตถุประสงค์การเรียนรู้

1. เข้าใจหลักการ Climate Models
2. ใช้ข้อมูล NetCDF
3. วิเคราะห์ข้อมูล Climate
4. สร้าง Visualizations

## โครงสร้างไฟล์

```
chapter-22-climate-modeling/
├── README.md
├── netcdf_basics.py        # Working with NetCDF
├── climate_analysis.py     # Climate data analysis
├── temperature_trends.py   # Temperature trend analysis
├── thai_rainfall.py        # Thailand rainfall analysis
└── sbatch/
    └── climate_job.sbatch
```

## การใช้งาน

```bash
# Create environment
mamba create -n hpc-climate python=3.9 netcdf4 xarray cartopy matplotlib
mamba activate hpc-climate

# Run examples
python netcdf_basics.py
python thai_rainfall.py
```

## Climate Data Sources

- ERA5: ECMWF Reanalysis
- CMIP6: Coupled Model Intercomparison Project
- Thai Meteorological Department
