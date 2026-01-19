#!/usr/bin/env python3
"""
MPI Collective Operations - การสื่อสารแบบกลุ่ม
Chapter 8: Advanced MPI Programming

แสดง: Broadcast, Scatter, Gather, Reduce, Allreduce
"""

import numpy as np
from mpi4py import MPI


def demo_broadcast(comm):
    """Broadcast: ส่งข้อมูลจาก root ไปยังทุก process"""
    rank = comm.Get_rank()

    if rank == 0:
        data = {"key": "value", "list": [1, 2, 3]}
        print(f"   Root (rank 0) broadcasting: {data}")
    else:
        data = None

    # Broadcast
    data = comm.bcast(data, root=0)
    print(f"   Rank {rank} received: {data}")


def demo_scatter(comm):
    """Scatter: กระจาย array ไปยังแต่ละ process"""
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        data = np.arange(size * 3)  # e.g., [0,1,2,3,4,5,6,7,8,9,10,11] for 4 processes
        print(f"   Root scattering: {data}")
        # Split into chunks
        chunks = np.array_split(data, size)
    else:
        chunks = None

    # Scatter
    local_data = comm.scatter(chunks, root=0)
    print(f"   Rank {rank} received: {local_data}")


def demo_gather(comm):
    """Gather: รวบรวม data จากทุก process"""
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Each process has local data
    local_data = np.array([rank * 10, rank * 10 + 1])
    print(f"   Rank {rank} sending: {local_data}")

    # Gather at root
    gathered = comm.gather(local_data, root=0)

    if rank == 0:
        print(f"   Root gathered: {gathered}")


def demo_reduce(comm):
    """Reduce: รวมค่าจากทุก process ด้วย operation"""
    rank = comm.Get_rank()

    local_value = rank + 1  # 1, 2, 3, 4, ...
    print(f"   Rank {rank} value: {local_value}")

    # Sum at root
    total = comm.reduce(local_value, op=MPI.SUM, root=0)

    if rank == 0:
        print(f"   Sum at root: {total}")


def demo_allreduce(comm):
    """Allreduce: reduce + broadcast ในคำสั่งเดียว"""
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_value = rank + 1
    print(f"   Rank {rank} value: {local_value}")

    # All processes get the result
    total = comm.allreduce(local_value, op=MPI.SUM)
    print(f"   Rank {rank} received total: {total}")


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("=" * 60)
        print("📡 MPI Collective Operations")
        print("   Chapter 8: Advanced MPI Programming")
        print("=" * 60)
        print(f"\n   Running with {size} processes")

    comm.Barrier()

    # Demo each collective
    demos = [
        ("Broadcast", demo_broadcast),
        ("Scatter", demo_scatter),
        ("Gather", demo_gather),
        ("Reduce", demo_reduce),
        ("Allreduce", demo_allreduce),
    ]

    for name, func in demos:
        comm.Barrier()
        if rank == 0:
            print(f"\n{'='*40}")
            print(f"📌 {name}")
            print(f"{'='*40}")
        comm.Barrier()
        func(comm)
        comm.Barrier()

    if rank == 0:
        print("\n" + "=" * 60)
        print("✅ Collective operations demo complete!")
        print("=" * 60)


if __name__ == "__main__":
    main()
