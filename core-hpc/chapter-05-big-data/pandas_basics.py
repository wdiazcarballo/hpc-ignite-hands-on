#!/usr/bin/env python3
"""
Pandas Basics - พื้นฐาน Pandas
Chapter 5: Big Data Processing

แสดง: DataFrame, Series, Groupby, Aggregation
"""

import numpy as np
import pandas as pd


def demonstrate_dataframe():
    """สาธิต DataFrame พื้นฐาน"""
    print("\n1. DataFrame Basics:")

    # Create from dictionary
    data = {
        'province': ['Chiang Mai', 'Chiang Rai', 'Lamphun', 'Lampang', 'Mae Hong Son'],
        'population': [1_800_000, 1_300_000, 400_000, 750_000, 280_000],
        'area_km2': [20_107, 11_678, 4_506, 12_534, 12_681],
        'rice_tons': [450_000, 380_000, 180_000, 220_000, 45_000]
    }

    df = pd.DataFrame(data)
    print(f"\n   Northern Thailand Provinces:")
    print(df.to_string(index=False))

    # Basic statistics
    print(f"\n   Total population: {df['population'].sum():,}")
    print(f"   Average rice production: {df['rice_tons'].mean():,.0f} tons")

    return df


def demonstrate_calculations(df):
    """สาธิตการคำนวณ"""
    print("\n2. Calculations:")

    # Add computed columns
    df['density'] = df['population'] / df['area_km2']
    df['rice_per_capita'] = df['rice_tons'] / df['population'] * 1000  # kg per person

    print(f"\n   With computed columns:")
    print(df[['province', 'density', 'rice_per_capita']].round(2).to_string(index=False))

    return df


def demonstrate_filtering(df):
    """สาธิตการกรองข้อมูล"""
    print("\n3. Filtering:")

    # Filter by condition
    large_provinces = df[df['population'] > 500_000]
    print(f"\n   Provinces with population > 500,000:")
    print(large_provinces[['province', 'population']].to_string(index=False))

    # Multiple conditions
    productive = df[(df['rice_tons'] > 200_000) & (df['area_km2'] > 10_000)]
    print(f"\n   Large & productive provinces:")
    print(productive[['province', 'rice_tons', 'area_km2']].to_string(index=False))


def demonstrate_groupby():
    """สาธิต Groupby operations"""
    print("\n4. Groupby Operations:")

    # Create data with categories
    data = {
        'district': ['Mueang', 'San Sai', 'Hang Dong', 'Mueang', 'Mae Rim'],
        'province': ['CM', 'CM', 'CM', 'CR', 'CM'],
        'crop': ['rice', 'longan', 'rice', 'rice', 'longan'],
        'yield_tons': [1200, 800, 950, 1100, 750]
    }
    df = pd.DataFrame(data)

    print(f"\n   Original data:")
    print(df.to_string(index=False))

    # Group by province
    by_province = df.groupby('province')['yield_tons'].sum()
    print(f"\n   Total yield by province:")
    print(by_province.to_string())

    # Group by crop
    by_crop = df.groupby('crop').agg({
        'yield_tons': ['sum', 'mean', 'count']
    })
    print(f"\n   Statistics by crop:")
    print(by_crop.to_string())


def demonstrate_merge():
    """สาธิตการ merge DataFrames"""
    print("\n5. Merging DataFrames:")

    # Production data
    production = pd.DataFrame({
        'province_id': [1, 2, 3],
        'rice_tons': [450_000, 380_000, 180_000]
    })

    # Province info
    provinces = pd.DataFrame({
        'province_id': [1, 2, 3],
        'name': ['Chiang Mai', 'Chiang Rai', 'Lamphun'],
        'region': ['North', 'North', 'North']
    })

    # Merge
    merged = pd.merge(production, provinces, on='province_id')
    print(f"\n   Merged data:")
    print(merged.to_string(index=False))


def main():
    print("=" * 60)
    print("   Pandas Basics")
    print("   Chapter 5: Big Data Processing")
    print("=" * 60)

    df = demonstrate_dataframe()
    df = demonstrate_calculations(df)
    demonstrate_filtering(df)
    demonstrate_groupby()
    demonstrate_merge()

    print("\n" + "=" * 60)
    print("   Pandas basics complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
