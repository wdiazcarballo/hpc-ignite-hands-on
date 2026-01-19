#!/usr/bin/env python3
"""
Chunk Processing - การประมวลผลแบบแบ่งส่วน
Chapter 5: Big Data Processing

ประมวลผลไฟล์ขนาดใหญ่โดยไม่โหลดทั้งหมดเข้า RAM
"""

import os
import time
import numpy as np
import pandas as pd


def generate_sample_data(filename: str, n_rows: int = 100_000):
    """สร้างไฟล์ข้อมูลตัวอย่าง"""
    print(f"\n   Generating {n_rows:,} rows of sample data...")

    np.random.seed(42)

    # Generate in chunks to avoid memory issues
    chunk_size = 10_000
    provinces = ['Chiang Mai', 'Chiang Rai', 'Lamphun', 'Lampang', 'Mae Hong Son']

    with open(filename, 'w') as f:
        f.write('date,province,temperature,humidity,rainfall\n')

        for i in range(0, n_rows, chunk_size):
            current_chunk = min(chunk_size, n_rows - i)

            dates = pd.date_range('2020-01-01', periods=current_chunk, freq='h')
            data = {
                'date': dates.strftime('%Y-%m-%d %H:%M'),
                'province': np.random.choice(provinces, current_chunk),
                'temperature': np.random.normal(28, 5, current_chunk).round(1),
                'humidity': np.random.uniform(40, 95, current_chunk).round(1),
                'rainfall': np.random.exponential(2, current_chunk).round(2)
            }

            for j in range(current_chunk):
                f.write(f"{data['date'][j]},{data['province'][j]},{data['temperature'][j]},{data['humidity'][j]},{data['rainfall'][j]}\n")

    file_size = os.path.getsize(filename) / (1024 * 1024)
    print(f"   Created: {filename} ({file_size:.1f} MB)")

    return filename


def process_without_chunks(filename: str):
    """ประมวลผลโดยโหลดทั้งหมด (ไม่แนะนำสำหรับไฟล์ใหญ่)"""
    print("\n1. Processing WITHOUT chunks (loads all into RAM):")

    start = time.time()
    df = pd.read_csv(filename)

    # Compute statistics
    result = df.groupby('province').agg({
        'temperature': 'mean',
        'humidity': 'mean',
        'rainfall': 'sum'
    }).round(2)

    elapsed = time.time() - start
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    print(f"   Memory used: {memory_mb:.1f} MB")
    print(f"   Time: {elapsed:.2f}s")
    print(f"\n   Results:")
    print(result.to_string())

    return result


def process_with_chunks(filename: str, chunk_size: int = 10_000):
    """ประมวลผลแบบ chunks (แนะนำสำหรับไฟล์ใหญ่)"""
    print(f"\n2. Processing WITH chunks (chunk_size={chunk_size:,}):")

    start = time.time()

    # Accumulators
    sum_temp = {}
    sum_humidity = {}
    sum_rainfall = {}
    count = {}

    chunk_num = 0
    for chunk in pd.read_csv(filename, chunksize=chunk_size):
        chunk_num += 1

        # Process each chunk
        for province in chunk['province'].unique():
            mask = chunk['province'] == province
            province_data = chunk[mask]

            if province not in count:
                sum_temp[province] = 0
                sum_humidity[province] = 0
                sum_rainfall[province] = 0
                count[province] = 0

            sum_temp[province] += province_data['temperature'].sum()
            sum_humidity[province] += province_data['humidity'].sum()
            sum_rainfall[province] += province_data['rainfall'].sum()
            count[province] += len(province_data)

    # Compute final statistics
    result = pd.DataFrame({
        'province': list(count.keys()),
        'temperature': [sum_temp[p] / count[p] for p in count.keys()],
        'humidity': [sum_humidity[p] / count[p] for p in count.keys()],
        'rainfall': [sum_rainfall[p] for p in count.keys()]
    }).set_index('province').round(2)

    elapsed = time.time() - start

    print(f"   Processed {chunk_num} chunks")
    print(f"   Peak memory: ~{chunk_size * 50 / (1024 * 1024):.1f} MB (much less!)")
    print(f"   Time: {elapsed:.2f}s")
    print(f"\n   Results:")
    print(result.to_string())

    return result


def demonstrate_iterator():
    """สาธิตการใช้ iterator"""
    print("\n3. Using Iterator for Line-by-Line Processing:")

    # Create small sample
    sample_file = '/tmp/small_sample.csv'
    with open(sample_file, 'w') as f:
        f.write('name,value\n')
        for i in range(5):
            f.write(f'item_{i},{i*10}\n')

    print("   Reading line by line:")
    total = 0
    with open(sample_file, 'r') as f:
        header = next(f)  # Skip header
        for line in f:
            parts = line.strip().split(',')
            name, value = parts[0], int(parts[1])
            total += value
            print(f"   {name}: {value}")

    print(f"   Total: {total}")


def main():
    print("=" * 60)
    print("   Chunk Processing")
    print("   Chapter 5: Big Data Processing")
    print("=" * 60)

    # Generate sample data
    filename = '/tmp/weather_data.csv'
    generate_sample_data(filename, n_rows=100_000)

    # Compare methods
    result1 = process_without_chunks(filename)
    result2 = process_with_chunks(filename, chunk_size=10_000)

    # Verify results match
    print("\n4. Verification:")
    if np.allclose(result1['temperature'].values, result2['temperature'].values, rtol=0.01):
        print("   Results match! Chunk processing gives same results.")

    demonstrate_iterator()

    # Cleanup
    os.remove(filename)

    print("\n" + "=" * 60)
    print("   Chunk processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
