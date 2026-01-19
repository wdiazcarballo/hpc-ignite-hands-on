#!/usr/bin/env python3
"""
MPI Hello World - ตัวอย่างแรกของ MPI
Chapter 3: Parallel Programming

การใช้งาน:
    mpirun -np 4 python hello_mpi.py
"""

from mpi4py import MPI


def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # Process ID (0 to size-1)
    size = comm.Get_size()  # Total number of processes
    name = MPI.Get_processor_name()  # Node name

    # Each process prints its info
    print(f"สวัสดีจาก Process {rank} of {size} บน {name}")

    # Synchronize all processes
    comm.Barrier()

    # Only rank 0 prints summary
    if rank == 0:
        print("\n" + "=" * 50)
        print(f"🎉 MPI ทำงานสำเร็จด้วย {size} processes!")
        print("=" * 50)


if __name__ == "__main__":
    main()
