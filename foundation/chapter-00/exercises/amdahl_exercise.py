#!/usr/bin/env python3
"""
แบบฝึกหัด: กฎของแอมดาห์ล (Amdahl's Law)
Chapter 0: HPC 101 - แบบฝึกหัดที่ 0.2

ให้นักศึกษาเติมโค้ดในส่วนที่มีเครื่องหมาย TODO
"""

import matplotlib.pyplot as plt
import numpy as np


def amdahl_speedup(P: float, N: int) -> float:
    """
    คำนวณ Speedup ตามกฎของแอมดาห์ล

    Parameters
    ----------
    P : float
        สัดส่วนของงานที่ทำแบบขนานได้ (0-1)
    N : int
        จำนวนหน่วยประมวลผล

    Returns
    -------
    float
        Speedup

    สูตร: S = 1 / ((1-P) + P/N)
    """
    # TODO: เติมสูตรของแอมดาห์ล
    # Hint: serial_fraction = 1 - P
    #       parallel_fraction = P / N
    #       speedup = 1 / (serial_fraction + parallel_fraction)

    pass  # ลบบรรทัดนี้แล้วเติมโค้ด


def main():
    """
    โปรแกรมหลัก:
    1. คำนวณ Speedup สำหรับ N = 2, 4, 8, 16, 32, 64, 128
    2. วาดกราฟความสัมพันธ์ระหว่าง N และ Speedup
    3. หา Speedup สูงสุดที่เป็นไปได้เมื่อ N → ∞
    """
    # ค่าที่กำหนด
    P = 0.95  # 95% parallel
    processors = [2, 4, 8, 16, 32, 64, 128]

    print("กฎของแอมดาห์ล (Amdahl's Law)")
    print("=" * 40)
    print(f"P (Parallel fraction): {P*100:.0f}%")
    print(f"Serial fraction: {(1-P)*100:.0f}%")
    print()

    # TODO: คำนวณ Speedup สำหรับแต่ละค่า N
    # speedups = [amdahl_speedup(P, n) for n in processors]

    # TODO: แสดงผลลัพธ์ในรูปแบบตาราง
    # print(f"{'N':<10} {'Speedup':<15}")
    # print("-" * 25)
    # for n, s in zip(processors, speedups):
    #     print(f"{n:<10} {s:<15.4f}")

    # TODO: คำนวณ Speedup สูงสุด (เมื่อ N → ∞)
    # max_speedup = 1 / (1 - P)
    # print(f"\nMax Speedup (N → ∞): {max_speedup:.2f}")

    # TODO: วาดกราฟ (ถ้าต้องการ)
    # plt.figure(figsize=(10, 6))
    # plt.plot(processors, speedups, 'b-o')
    # plt.xlabel('Number of Processors')
    # plt.ylabel('Speedup')
    # plt.title(f"Amdahl's Law (P = {P*100:.0f}%)")
    # plt.grid(True)
    # plt.show()


if __name__ == "__main__":
    main()
