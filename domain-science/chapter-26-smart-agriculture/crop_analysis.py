#!/usr/bin/env python3
"""
Crop Analysis - การวิเคราะห์ผลผลิตพืช
Chapter 26: Smart Agriculture

วิเคราะห์ผลผลิตทางการเกษตรของภาคเหนือ
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def generate_crop_data():
    """สร้างข้อมูลจำลองผลผลิตข้าวภาคเหนือ"""
    print("\n1. Generating Northern Thailand Rice Yield Data:")

    np.random.seed(42)

    provinces = ['Chiang Mai', 'Chiang Rai', 'Lamphun', 'Lampang', 'Mae Hong Son']
    years = range(2000, 2024)

    data = []
    for province in provinces:
        base_yield = {'Chiang Mai': 450, 'Chiang Rai': 420, 'Lamphun': 380,
                      'Lampang': 400, 'Mae Hong Son': 350}[province]

        for year in years:
            # Weather factors
            rainfall = np.random.normal(1200, 200)  # mm
            temp_avg = np.random.normal(26, 1.5)  # °C
            drought_days = max(0, np.random.normal(15, 10))

            # Yield affected by weather and technology improvement
            tech_factor = 1 + 0.01 * (year - 2000)  # 1% per year
            rain_factor = min(1.2, rainfall / 1000) if rainfall > 800 else rainfall / 1000
            temp_factor = 1 - abs(temp_avg - 26) * 0.02
            drought_factor = 1 - drought_days * 0.01

            yield_kg = base_yield * tech_factor * rain_factor * temp_factor * drought_factor
            yield_kg += np.random.normal(0, 30)  # Random variation
            yield_kg = max(200, yield_kg)  # Minimum yield

            data.append({
                'province': province,
                'year': year,
                'rainfall_mm': rainfall,
                'temp_avg_c': temp_avg,
                'drought_days': drought_days,
                'yield_kg_rai': yield_kg,  # kg per rai (Thai unit)
                'yield_ton_ha': yield_kg * 6.25 / 1000  # Convert to ton/ha
            })

    df = pd.DataFrame(data)
    print(f"   Records: {len(df)}")
    print(f"   Years: {min(years)}-{max(years)}")
    print(f"   Provinces: {len(provinces)}")

    return df


def analyze_yield_trends(df):
    """วิเคราะห์แนวโน้มผลผลิต"""
    print("\n2. Yield Trend Analysis:")

    # Overall trend
    annual_avg = df.groupby('year')['yield_kg_rai'].mean()

    # Linear regression for trend
    X = np.array(annual_avg.index).reshape(-1, 1)
    y = annual_avg.values
    model = LinearRegression()
    model.fit(X, y)

    trend_per_year = model.coef_[0]
    print(f"   Average yield trend: {trend_per_year:+.2f} kg/rai per year")
    print(f"   Yield 2000: {annual_avg.iloc[0]:.0f} kg/rai")
    print(f"   Yield 2023: {annual_avg.iloc[-1]:.0f} kg/rai")
    print(f"   Total improvement: {annual_avg.iloc[-1] - annual_avg.iloc[0]:.0f} kg/rai ({(annual_avg.iloc[-1]/annual_avg.iloc[0] - 1)*100:.1f}%)")

    # By province
    print(f"\n   Average yield by province (2020-2023):")
    recent = df[df['year'] >= 2020]
    by_province = recent.groupby('province')['yield_kg_rai'].mean().sort_values(ascending=False)
    for province, yield_val in by_province.items():
        print(f"   {province}: {yield_val:.0f} kg/rai")


def analyze_weather_correlation(df):
    """วิเคราะห์ความสัมพันธ์กับสภาพอากาศ"""
    print("\n3. Weather Correlation Analysis:")

    # Correlation matrix
    weather_cols = ['rainfall_mm', 'temp_avg_c', 'drought_days']
    correlations = df[weather_cols + ['yield_kg_rai']].corr()['yield_kg_rai'].drop('yield_kg_rai')

    print(f"\n   Correlation with yield:")
    for col, corr in correlations.items():
        direction = "positive" if corr > 0 else "negative"
        strength = "strong" if abs(corr) > 0.5 else "moderate" if abs(corr) > 0.3 else "weak"
        print(f"   {col}: {corr:+.3f} ({strength} {direction})")


def build_yield_model(df):
    """สร้างโมเดลทำนายผลผลิต"""
    print("\n4. Building Yield Prediction Model:")

    # Features
    features = ['rainfall_mm', 'temp_avg_c', 'drought_days', 'year']

    # One-hot encode province
    df_model = pd.get_dummies(df, columns=['province'])
    feature_cols = features + [col for col in df_model.columns if col.startswith('province_')]

    X = df_model[feature_cols]
    y = df_model['yield_kg_rai']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"   Model: Random Forest (100 trees)")
    print(f"   Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"   RMSE: {rmse:.1f} kg/rai")
    print(f"   R²: {r2:.3f}")

    # Feature importance
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    top_features = importance.nlargest(5)
    print(f"\n   Top 5 features:")
    for feat, imp in top_features.items():
        print(f"   {feat}: {imp:.3f}")

    return model


def plot_analysis(df, save_file='crop_analysis.png'):
    """สร้างกราฟวิเคราะห์"""
    print("\n5. Generating Visualizations:")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Yield trends by province
    ax1 = axes[0, 0]
    for province in df['province'].unique():
        prov_data = df[df['province'] == province]
        annual = prov_data.groupby('year')['yield_kg_rai'].mean()
        ax1.plot(annual.index, annual.values, label=province, linewidth=2, marker='o', markersize=3)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Yield (kg/rai)')
    ax1.set_title('Rice Yield Trends by Province')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Rainfall vs Yield scatter
    ax2 = axes[0, 1]
    scatter = ax2.scatter(df['rainfall_mm'], df['yield_kg_rai'],
                          c=df['year'], cmap='viridis', alpha=0.6, s=30)
    ax2.set_xlabel('Rainfall (mm)')
    ax2.set_ylabel('Yield (kg/rai)')
    ax2.set_title('Rainfall vs Yield')
    plt.colorbar(scatter, ax=ax2, label='Year')

    # 3. Temperature distribution
    ax3 = axes[1, 0]
    ax3.hist2d(df['temp_avg_c'], df['yield_kg_rai'], bins=20, cmap='YlOrRd')
    ax3.set_xlabel('Temperature (°C)')
    ax3.set_ylabel('Yield (kg/rai)')
    ax3.set_title('Temperature vs Yield Density')

    # 4. Box plot by province
    ax4 = axes[1, 1]
    df.boxplot(column='yield_kg_rai', by='province', ax=ax4)
    ax4.set_xlabel('Province')
    ax4.set_ylabel('Yield (kg/rai)')
    ax4.set_title('Yield Distribution by Province')
    plt.suptitle('')  # Remove auto title

    plt.suptitle('Northern Thailand Rice Production Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    print(f"   Saved: {save_file}")
    plt.close()


def main():
    print("=" * 60)
    print("   Crop Analysis - Northern Thailand Rice")
    print("   Chapter 26: Smart Agriculture")
    print("=" * 60)

    # Generate data
    df = generate_crop_data()

    # Analysis
    analyze_yield_trends(df)
    analyze_weather_correlation(df)

    # Build model
    model = build_yield_model(df)

    # Visualization
    plot_analysis(df)

    print("\n" + "=" * 60)
    print("   Crop analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
