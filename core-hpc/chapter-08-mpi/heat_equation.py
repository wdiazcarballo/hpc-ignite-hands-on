#!/usr/bin/env python3
"""
Heat Equation Solver - การจำลองการแพร่กระจายความร้อน
Chapter 8: Advanced MPI Programming

แก้สมการความร้อน 1D: ∂u/∂t = α * ∂²u/∂x²
ด้วย Domain Decomposition และ Ghost Cell Exchange
"""

import argparse
import numpy as np
from mpi4py import MPI


def initialize_domain(n_local, rank, size):
    """สร้าง initial condition"""
    # Global domain [0, 1], initial condition: u(x,0) = sin(πx)
    dx = 1.0 / (n_local * size - 1)
    start_idx = rank * n_local
    x_local = np.linspace(start_idx * dx, (start_idx + n_local - 1) * dx, n_local)

    # Initial temperature: hot in the middle
    u = np.sin(np.pi * x_local)

    return u, dx


def exchange_ghosts(u, comm, rank, size):
    """แลกเปลี่ยน ghost cells กับ neighbors"""
    # Send right, receive left
    if rank < size - 1:
        comm.send(u[-1], dest=rank + 1, tag=0)
    if rank > 0:
        left_ghost = comm.recv(source=rank - 1, tag=0)
    else:
        left_ghost = 0.0  # Boundary condition

    # Send left, receive right
    if rank > 0:
        comm.send(u[0], dest=rank - 1, tag=1)
    if rank < size - 1:
        right_ghost = comm.recv(source=rank + 1, tag=1)
    else:
        right_ghost = 0.0  # Boundary condition

    return left_ghost, right_ghost


def update_temperature(u, left_ghost, right_ghost, alpha, dt, dx):
    """อัพเดตอุณหภูมิด้วย explicit finite difference"""
    n = len(u)
    u_new = np.zeros_like(u)

    # Coefficient
    r = alpha * dt / (dx * dx)

    # Update interior points
    for i in range(1, n - 1):
        u_new[i] = u[i] + r * (u[i+1] - 2*u[i] + u[i-1])

    # Update boundary points using ghosts
    u_new[0] = u[0] + r * (u[1] - 2*u[0] + left_ghost)
    u_new[-1] = u[-1] + r * (right_ghost - 2*u[-1] + u[-2])

    return u_new


def main():
    parser = argparse.ArgumentParser(description="1D Heat Equation with MPI")
    parser.add_argument("--n", type=int, default=100, help="Points per process")
    parser.add_argument("--steps", type=int, default=1000, help="Time steps")
    parser.add_argument("--alpha", type=float, default=0.01, help="Diffusion coefficient")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Initialize
    u, dx = initialize_domain(args.n, rank, size)
    dt = 0.4 * dx * dx / args.alpha  # CFL condition

    if rank == 0:
        print("=" * 60)
        print("🔥 1D Heat Equation Solver with MPI")
        print("   Chapter 8: Advanced MPI Programming")
        print("=" * 60)
        print(f"\n   Processes: {size}")
        print(f"   Points per process: {args.n}")
        print(f"   Total points: {args.n * size}")
        print(f"   Time steps: {args.steps}")
        print(f"   dt: {dt:.6f}, dx: {dx:.6f}")

    # Time stepping
    start_time = MPI.Wtime()

    for step in range(args.steps):
        # Exchange ghost cells
        left_ghost, right_ghost = exchange_ghosts(u, comm, rank, size)

        # Update temperature
        u = update_temperature(u, left_ghost, right_ghost, args.alpha, dt, dx)

        # Print progress
        if rank == 0 and (step + 1) % 200 == 0:
            # Compute global max temperature
            local_max = np.max(u)
            global_max = comm.reduce(local_max, op=MPI.MAX, root=0)
            print(f"   Step {step+1:4d}: max temperature = {global_max:.6f}")

    elapsed = MPI.Wtime() - start_time

    # Final statistics
    local_sum = np.sum(u)
    global_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

    if rank == 0:
        avg_temp = global_sum / (args.n * size)
        print(f"\n   Final average temperature: {avg_temp:.6f}")
        print(f"   Elapsed time: {elapsed:.3f}s")
        print(f"   Performance: {args.steps * args.n * size / elapsed / 1e6:.2f} million updates/s")
        print("\n" + "=" * 60)
        print("✅ Heat equation simulation complete!")
        print("=" * 60)


if __name__ == "__main__":
    main()
