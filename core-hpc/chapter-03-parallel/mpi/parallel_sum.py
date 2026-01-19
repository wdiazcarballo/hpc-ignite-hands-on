#!/usr/bin/env python3
"""
Parallel Sum - การรวมผลแบบขนานด้วย MPI
Chapter 3: Parallel Programming

คำนวณผลรวม 1 + 2 + ... + N โดยแบ่งงานให้แต่ละ process

การใช้งาน:
    mpirun -np 4 python parallel_sum.py
    mpirun -np 8 python parallel_sum.py --n 10000000
"""

import argparse
import time
from mpi4py import MPI


def parallel_sum(N: int, comm) -> tuple:
    """
    คำนวณ 1 + 2 + ... + N แบบขนาน

    Parameters
    ----------
    N : int
        จำนวนที่ต้องการรวม (1 ถึง N)
    comm : MPI.Comm
        MPI communicator

    Returns
    -------
    tuple
        (total_sum, elapsed_time) สำหรับ rank 0
        (None, elapsed_time) สำหรับ rank อื่น
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Start timing
    start_time = MPI.Wtime()

    # แบ่งงาน: แต่ละ process คำนวณส่วนของตัวเอง
    chunk_size = N // size
    remainder = N % size

    # กระจายงานให้เท่ากัน (พยายาม)
    if rank < remainder:
        start = rank * (chunk_size + 1) + 1
        end = start + chunk_size + 1
    else:
        start = rank * chunk_size + remainder + 1
        end = start + chunk_size

    # คำนวณผลรวมเฉพาะส่วน (local sum)
    local_sum = sum(range(start, end))

    if rank == 0:
        print(f"Process {rank}: calculating sum({start} to {end-1})")

    # รวมผลจากทุก process (reduce operation)
    total_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

    # End timing
    elapsed = MPI.Wtime() - start_time

    return total_sum, elapsed


def sequential_sum(N: int) -> tuple:
    """คำนวณแบบลำดับ (สำหรับเปรียบเทียบ)"""
    start_time = time.time()
    result = sum(range(1, N + 1))
    elapsed = time.time() - start_time
    return result, elapsed


def main():
    parser = argparse.ArgumentParser(description="Parallel Sum with MPI")
    parser.add_argument("--n", type=int, default=1_000_000, help="Sum from 1 to N")
    args = parser.parse_args()

    N = args.n
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("=" * 60)
        print("🔢 Parallel Sum with MPI")
        print("   Chapter 3: Parallel Programming")
        print("=" * 60)
        print(f"\nCalculating: 1 + 2 + ... + {N:,}")
        print(f"Using {size} MPI processes\n")

    # Parallel calculation
    total_sum, parallel_time = parallel_sum(N, comm)

    if rank == 0:
        # Verify with formula: n(n+1)/2
        expected = N * (N + 1) // 2

        print("\n" + "-" * 40)
        print("📊 Results:")
        print(f"   Parallel sum: {total_sum:,}")
        print(f"   Expected (formula): {expected:,}")
        print(f"   Match: {'✅ Yes' if total_sum == expected else '❌ No'}")
        print(f"\n⏱️ Parallel time: {parallel_time:.6f} seconds")

        # Sequential comparison (only for small N)
        if N <= 10_000_000:
            seq_sum, seq_time = sequential_sum(N)
            speedup = seq_time / parallel_time if parallel_time > 0 else float("inf")
            print(f"⏱️ Sequential time: {seq_time:.6f} seconds")
            print(f"🚀 Speedup: {speedup:.2f}x")

        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
