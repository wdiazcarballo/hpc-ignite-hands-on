#!/usr/bin/env python3
"""
Monte Carlo Pi Estimation - การประมาณค่า π ด้วยวิธี Monte Carlo
Chapter 3: Parallel Programming

หลักการ: สุ่มจุดในสี่เหลี่ยมจัตุรัส แล้วนับจุดที่ตกในวงกลม
π ≈ 4 × (จุดในวงกลม / จุดทั้งหมด)

การใช้งาน:
    mpirun -np 4 python monte_carlo_pi.py
    mpirun -np 8 python monte_carlo_pi.py --samples 100000000
"""

import argparse
import numpy as np
from mpi4py import MPI


def monte_carlo_pi_local(n_samples: int, seed: int = None) -> int:
    """
    นับจุดที่ตกในวงกลมรัศมี 1 (local calculation)

    Parameters
    ----------
    n_samples : int
        จำนวนจุดที่จะสุ่ม
    seed : int, optional
        Random seed (ต่าง process ต่าง seed)

    Returns
    -------
    int
        จำนวนจุดที่ตกในวงกลม
    """
    if seed is not None:
        np.random.seed(seed)

    # สุ่มจุด (x, y) ในช่วง [0, 1]
    x = np.random.random(n_samples)
    y = np.random.random(n_samples)

    # นับจุดที่อยู่ในวงกลม (x² + y² ≤ 1)
    inside_circle = np.sum(x**2 + y**2 <= 1)

    return inside_circle


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo Pi Estimation")
    parser.add_argument("--samples", type=int, default=10_000_000,
                       help="Total number of samples")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    total_samples = args.samples
    samples_per_process = total_samples // size

    if rank == 0:
        print("=" * 60)
        print("🎯 Monte Carlo Pi Estimation with MPI")
        print("   Chapter 3: Parallel Programming")
        print("=" * 60)
        print(f"\nTotal samples: {total_samples:,}")
        print(f"Processes: {size}")
        print(f"Samples per process: {samples_per_process:,}\n")

    # Start timing
    start_time = MPI.Wtime()

    # Each process calculates with different seed
    local_inside = monte_carlo_pi_local(samples_per_process, seed=rank * 12345)

    # Gather results
    total_inside = comm.reduce(local_inside, op=MPI.SUM, root=0)

    # End timing
    elapsed = MPI.Wtime() - start_time

    if rank == 0:
        # Calculate pi
        pi_estimate = 4.0 * total_inside / (samples_per_process * size)
        error = abs(pi_estimate - np.pi)
        error_percent = error / np.pi * 100

        print("-" * 40)
        print("📊 Results:")
        print(f"   Points inside circle: {total_inside:,}")
        print(f"   Total points: {samples_per_process * size:,}")
        print(f"\n   π estimate: {pi_estimate:.10f}")
        print(f"   π actual:   {np.pi:.10f}")
        print(f"   Error: {error:.10f} ({error_percent:.6f}%)")
        print(f"\n⏱️ Time: {elapsed:.4f} seconds")
        print(f"🚀 Throughput: {total_samples/elapsed/1e6:.2f} million samples/sec")
        print("=" * 60)


if __name__ == "__main__":
    main()
