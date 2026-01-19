#!/usr/bin/env python3
"""
กฎของแอมดาห์ล (Amdahl's Law) - การคำนวณ Speedup
Chapter 0: HPC 101 - แบบฝึกหัดระดับอุดมศึกษา

สูตร: S = 1 / ((1-P) + P/N)
- S = Speedup
- P = สัดส่วนของงานที่ทำแบบขนานได้
- N = จำนวนหน่วยประมวลผล
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
        Speedup (ความเร็วที่เพิ่มขึ้น)

    Examples
    --------
    >>> amdahl_speedup(0.9, 10)
    5.263157894736842
    >>> amdahl_speedup(0.95, 100)
    16.806722689075632
    """
    serial_fraction = 1 - P
    parallel_fraction = P / N
    speedup = 1 / (serial_fraction + parallel_fraction)
    return speedup


def max_speedup(P: float) -> float:
    """
    คำนวณ Speedup สูงสุดที่เป็นไปได้ (เมื่อ N → ∞)

    Parameters
    ----------
    P : float
        สัดส่วนของงานที่ทำแบบขนานได้

    Returns
    -------
    float
        Speedup สูงสุด = 1 / (1-P)
    """
    return 1 / (1 - P)


def plot_speedup_curve(P: float, processors: list, save_path: str = None):
    """
    วาดกราฟ Speedup vs จำนวน Processors

    Parameters
    ----------
    P : float
        สัดส่วนของงานที่ทำแบบขนานได้
    processors : list
        รายการจำนวน processors ที่ต้องการทดสอบ
    save_path : str, optional
        Path สำหรับบันทึกรูป
    """
    speedups = [amdahl_speedup(P, n) for n in processors]
    max_s = max_speedup(P)

    plt.figure(figsize=(10, 6))

    # Plot speedup curve
    plt.plot(processors, speedups, "b-o", linewidth=2, markersize=8, label="Actual Speedup")

    # Plot max speedup line
    plt.axhline(y=max_s, color="r", linestyle="--", linewidth=2, label=f"Max Speedup = {max_s:.2f}")

    # Plot ideal speedup (linear)
    plt.plot(processors, processors, "g:", linewidth=2, label="Ideal (Linear) Speedup")

    plt.xlabel("Number of Processors (N)", fontsize=12)
    plt.ylabel("Speedup (S)", fontsize=12)
    plt.title(f"Amdahl's Law: Speedup vs Processors (P = {P*100:.0f}% parallel)", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Set axis limits
    plt.xlim(0, max(processors) * 1.1)
    plt.ylim(0, max(max_s, max(processors)) * 1.1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"📊 Graph saved to: {save_path}")

    plt.show()


def main():
    print("=" * 60)
    print("📐 กฎของแอมดาห์ล (Amdahl's Law)")
    print("   Chapter 0: HPC 101 - Understanding Parallel Speedup")
    print("=" * 60)

    # ค่าที่กำหนด
    P = 0.95  # 95% ของงานทำแบบขนานได้
    processors = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    print(f"\n📋 Input Parameters:")
    print(f"   P (Parallel fraction): {P*100:.0f}%")
    print(f"   Serial fraction: {(1-P)*100:.0f}%")

    print(f"\n📊 Speedup Calculations:")
    print("-" * 40)
    print(f"{'Processors (N)':<15} {'Speedup (S)':<15} {'Efficiency':<15}")
    print("-" * 40)

    for n in processors:
        s = amdahl_speedup(P, n)
        efficiency = s / n * 100
        print(f"{n:<15} {s:<15.4f} {efficiency:<15.2f}%")

    max_s = max_speedup(P)
    print("-" * 40)
    print(f"\n🎯 Maximum Speedup (N → ∞): {max_s:.2f}x")
    print(f"   ซึ่งหมายความว่าแม้ใช้ processors จำนวนมากเท่าใด")
    print(f"   ก็ไม่สามารถเร็วขึ้นเกิน {max_s:.2f} เท่าได้")
    print(f"   เพราะยังมีส่วน {(1-P)*100:.0f}% ที่ต้องทำแบบลำดับ")

    print("\n" + "=" * 60)
    print("💡 บทเรียนสำคัญ:")
    print("   1. การเพิ่ม processors มีจุดที่ไม่คุ้มค่า (diminishing returns)")
    print("   2. ส่วนที่ทำแบบลำดับเป็นคอขวดหลัก")
    print("   3. ควรพยายามลดส่วน serial ให้น้อยที่สุด")
    print("=" * 60)

    # วาดกราฟ
    try:
        plot_speedup_curve(P, processors)
    except Exception as e:
        print(f"\n⚠️ ไม่สามารถวาดกราฟได้: {e}")
        print("   (ต้องรันในสภาพแวดล้อมที่มี display)")


if __name__ == "__main__":
    main()
