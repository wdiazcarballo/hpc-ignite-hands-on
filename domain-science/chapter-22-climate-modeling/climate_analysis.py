#!/usr/bin/env python3
"""
Climate Analysis - การวิเคราะห์ภูมิอากาศ
Chapter 22: Climate Modeling

วิเคราะห์ข้อมูลอุณหภูมิและปริมาณน้ำฝนของภาคเหนือประเทศไทย
"""

import numpy as np
import matplotlib.pyplot as plt

# Check for optional packages
try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False


def generate_sample_data():
    """สร้างข้อมูลตัวอย่างจำลองภูมิอากาศภาคเหนือ"""
    print("\n1. Generating Sample Climate Data:")
    print("   (Simulated Northern Thailand climate data)")

    np.random.seed(42)

    # Time series: 30 years, monthly
    years = np.arange(1990, 2020)
    months = np.arange(1, 13)

    # Northern Thailand stations
    stations = {
        'Chiang Mai': {'lat': 18.79, 'lon': 98.98, 'elev': 310},
        'Chiang Rai': {'lat': 19.91, 'lon': 99.83, 'elev': 393},
        'Lamphun': {'lat': 18.58, 'lon': 99.00, 'elev': 289},
        'Mae Hong Son': {'lat': 19.30, 'lon': 97.97, 'elev': 268},
    }

    data = {}
    for station, info in stations.items():
        n_years = len(years)

        # Monthly temperature pattern (seasonal)
        monthly_temp_pattern = np.array([
            21, 23, 26, 28, 28, 27, 26, 26, 26, 25, 23, 21
        ])

        # Temperature with trend and noise
        temp = []
        for year_idx, year in enumerate(years):
            for month in months:
                base_temp = monthly_temp_pattern[month - 1]
                trend = 0.03 * year_idx  # Warming trend
                noise = np.random.normal(0, 1.5)
                temp.append(base_temp + trend + noise)

        # Rainfall pattern (monsoon)
        monthly_rain_pattern = np.array([
            10, 5, 20, 50, 150, 180, 200, 220, 180, 100, 40, 15
        ])

        rain = []
        for year_idx, year in enumerate(years):
            for month in months:
                base_rain = monthly_rain_pattern[month - 1]
                noise = np.random.exponential(base_rain * 0.3)
                rain.append(max(0, base_rain + noise - base_rain * 0.1))

        data[station] = {
            'temperature': np.array(temp),
            'rainfall': np.array(rain),
            'years': np.repeat(years, 12),
            'months': np.tile(months, n_years),
            **info
        }

    print(f"   Generated data for {len(stations)} stations")
    print(f"   Time period: {years[0]}-{years[-1]}")

    return data


def analyze_temperature_trends(data):
    """วิเคราะห์แนวโน้มอุณหภูมิ"""
    print("\n2. Temperature Trend Analysis:")

    for station, station_data in data.items():
        temp = station_data['temperature']
        years = station_data['years']

        # Calculate annual averages
        unique_years = np.unique(years)
        annual_avg = [temp[years == y].mean() for y in unique_years]

        # Linear regression
        slope, intercept = np.polyfit(unique_years, annual_avg, 1)
        trend_per_decade = slope * 10

        print(f"\n   {station}:")
        print(f"      Mean temperature: {np.mean(temp):.1f}°C")
        print(f"      Trend: {trend_per_decade:+.2f}°C per decade")
        print(f"      Max recorded: {np.max(temp):.1f}°C")
        print(f"      Min recorded: {np.min(temp):.1f}°C")


def analyze_rainfall_patterns(data):
    """วิเคราะห์รูปแบบปริมาณน้ำฝน"""
    print("\n3. Rainfall Pattern Analysis:")

    for station, station_data in data.items():
        rain = station_data['rainfall']
        months = station_data['months']
        years = station_data['years']

        # Monthly averages
        monthly_avg = [rain[months == m].mean() for m in range(1, 13)]
        annual_total = sum(monthly_avg)

        # Find wet/dry season
        wet_months = np.array(monthly_avg) > annual_total / 12
        wet_season = [i+1 for i, w in enumerate(wet_months) if w]

        print(f"\n   {station}:")
        print(f"      Annual rainfall: {annual_total:.0f} mm")
        print(f"      Wet season months: {wet_season}")
        print(f"      Peak month: {np.argmax(monthly_avg) + 1} ({max(monthly_avg):.0f} mm)")
        print(f"      Driest month: {np.argmin(monthly_avg) + 1} ({min(monthly_avg):.0f} mm)")


def plot_climate_summary(data, save_file='climate_summary.png'):
    """สร้างกราฟสรุปภูมิอากาศ"""
    print("\n4. Generating Climate Summary Plot:")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Colors for stations
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    stations = list(data.keys())

    # 1. Annual temperature trend
    ax1 = axes[0, 0]
    for i, (station, station_data) in enumerate(data.items()):
        years = station_data['years']
        temp = station_data['temperature']
        unique_years = np.unique(years)
        annual_avg = [temp[years == y].mean() for y in unique_years]
        ax1.plot(unique_years, annual_avg, color=colors[i], label=station, linewidth=1.5)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Annual Average Temperature')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Monthly temperature pattern
    ax2 = axes[0, 1]
    month_names = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    for i, (station, station_data) in enumerate(data.items()):
        months = station_data['months']
        temp = station_data['temperature']
        monthly_avg = [temp[months == m].mean() for m in range(1, 13)]
        ax2.plot(range(1, 13), monthly_avg, color=colors[i], label=station,
                 linewidth=2, marker='o', markersize=4)
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Temperature (°C)')
    ax2.set_title('Monthly Temperature Pattern')
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(month_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Monthly rainfall pattern
    ax3 = axes[1, 0]
    bar_width = 0.2
    x = np.arange(12)
    for i, (station, station_data) in enumerate(data.items()):
        months = station_data['months']
        rain = station_data['rainfall']
        monthly_avg = [rain[months == m].mean() for m in range(1, 13)]
        ax3.bar(x + i * bar_width, monthly_avg, bar_width, color=colors[i],
                label=station, alpha=0.8)
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Rainfall (mm)')
    ax3.set_title('Monthly Rainfall Pattern')
    ax3.set_xticks(x + bar_width * 1.5)
    ax3.set_xticklabels(month_names)
    ax3.legend()

    # 4. Temperature vs Rainfall scatter
    ax4 = axes[1, 1]
    for i, (station, station_data) in enumerate(data.items()):
        months = station_data['months']
        temp = station_data['temperature']
        rain = station_data['rainfall']
        monthly_temp = [temp[months == m].mean() for m in range(1, 13)]
        monthly_rain = [rain[months == m].mean() for m in range(1, 13)]
        ax4.scatter(monthly_temp, monthly_rain, color=colors[i], label=station,
                    s=80, alpha=0.7)
    ax4.set_xlabel('Temperature (°C)')
    ax4.set_ylabel('Rainfall (mm)')
    ax4.set_title('Temperature vs Rainfall')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Northern Thailand Climate Summary (1990-2019)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    print(f"   Saved: {save_file}")
    plt.close()


def main():
    print("=" * 60)
    print("   Climate Analysis")
    print("   Chapter 22: Climate Modeling")
    print("=" * 60)

    # Generate sample data
    data = generate_sample_data()

    # Analysis
    analyze_temperature_trends(data)
    analyze_rainfall_patterns(data)

    # Visualization
    plot_climate_summary(data)

    print("\n" + "=" * 60)
    print("   Climate analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
