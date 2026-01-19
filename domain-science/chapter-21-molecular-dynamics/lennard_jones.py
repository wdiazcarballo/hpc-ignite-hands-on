#!/usr/bin/env python3
"""
Lennard-Jones Simulation - การจำลอง LJ
Chapter 21: Molecular Dynamics

Simple 2D Lennard-Jones simulation for educational purposes
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


class LennardJonesMD:
    """Simple 2D Lennard-Jones Molecular Dynamics"""

    def __init__(self, n_particles: int = 36, box_size: float = 10.0,
                 epsilon: float = 1.0, sigma: float = 1.0,
                 temperature: float = 1.0, dt: float = 0.005):
        self.n = n_particles
        self.L = box_size
        self.eps = epsilon
        self.sig = sigma
        self.T = temperature
        self.dt = dt

        # Initialize on grid
        n_side = int(np.sqrt(n_particles))
        spacing = box_size / (n_side + 1)
        self.positions = np.zeros((n_particles, 2))

        idx = 0
        for i in range(n_side):
            for j in range(n_side):
                if idx < n_particles:
                    self.positions[idx] = [(i + 1) * spacing, (j + 1) * spacing]
                    idx += 1

        # Random velocities (Maxwell-Boltzmann)
        self.velocities = np.random.randn(n_particles, 2) * np.sqrt(temperature)

        # Remove center of mass velocity
        self.velocities -= self.velocities.mean(axis=0)

        self.forces = np.zeros((n_particles, 2))
        self.potential_energy = 0.0

    def compute_forces(self) -> float:
        """Calculate LJ forces and potential energy"""
        self.forces.fill(0.0)
        self.potential_energy = 0.0

        for i in range(self.n):
            for j in range(i + 1, self.n):
                # Distance with periodic boundary conditions
                dr = self.positions[j] - self.positions[i]
                dr = dr - self.L * np.round(dr / self.L)
                r2 = np.sum(dr ** 2)
                r = np.sqrt(r2)

                # LJ cutoff
                if r < 2.5 * self.sig:
                    # LJ potential: 4*eps*((sig/r)^12 - (sig/r)^6)
                    sig_r6 = (self.sig / r) ** 6
                    sig_r12 = sig_r6 ** 2

                    # Force magnitude
                    f_mag = 24 * self.eps / r * (2 * sig_r12 - sig_r6)

                    # Force vector
                    f = f_mag * dr / r
                    self.forces[i] -= f
                    self.forces[j] += f

                    # Potential energy
                    self.potential_energy += 4 * self.eps * (sig_r12 - sig_r6)

        return self.potential_energy

    def velocity_verlet_step(self):
        """One step of velocity Verlet integration"""
        # Half-step velocity update
        self.velocities += 0.5 * self.dt * self.forces

        # Full position update
        self.positions += self.dt * self.velocities

        # Apply periodic boundaries
        self.positions = self.positions % self.L

        # Compute new forces
        self.compute_forces()

        # Complete velocity update
        self.velocities += 0.5 * self.dt * self.forces

    def kinetic_energy(self) -> float:
        """Calculate kinetic energy"""
        return 0.5 * np.sum(self.velocities ** 2)

    def temperature_actual(self) -> float:
        """Calculate actual temperature from kinetic energy"""
        return 2 * self.kinetic_energy() / (2 * self.n - 2)

    def run_simulation(self, n_steps: int = 1000, save_every: int = 10):
        """Run MD simulation"""
        # Initial force calculation
        self.compute_forces()

        # Storage for trajectory and energies
        n_save = n_steps // save_every + 1
        trajectory = np.zeros((n_save, self.n, 2))
        energies = {'potential': [], 'kinetic': [], 'total': [], 'temperature': []}

        save_idx = 0
        for step in range(n_steps):
            self.velocity_verlet_step()

            if step % save_every == 0:
                trajectory[save_idx] = self.positions.copy()
                pe = self.potential_energy
                ke = self.kinetic_energy()
                temp = self.temperature_actual()

                energies['potential'].append(pe)
                energies['kinetic'].append(ke)
                energies['total'].append(pe + ke)
                energies['temperature'].append(temp)

                save_idx += 1

        return trajectory, energies


def plot_results(trajectory, energies, save_file='lj_simulation.png'):
    """Plot simulation results"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Initial configuration
    ax1 = axes[0, 0]
    ax1.scatter(trajectory[0, :, 0], trajectory[0, :, 1], s=100, alpha=0.7)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_aspect('equal')
    ax1.set_title('Initial Configuration')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    # Final configuration
    ax2 = axes[0, 1]
    ax2.scatter(trajectory[-1, :, 0], trajectory[-1, :, 1], s=100, alpha=0.7)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_aspect('equal')
    ax2.set_title('Final Configuration')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    # Energy
    ax3 = axes[1, 0]
    steps = range(len(energies['total']))
    ax3.plot(steps, energies['total'], label='Total', linewidth=2)
    ax3.plot(steps, energies['kinetic'], label='Kinetic', linewidth=1, alpha=0.7)
    ax3.plot(steps, energies['potential'], label='Potential', linewidth=1, alpha=0.7)
    ax3.set_xlabel('Step (x10)')
    ax3.set_ylabel('Energy')
    ax3.set_title('Energy Conservation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Temperature
    ax4 = axes[1, 1]
    ax4.plot(steps, energies['temperature'], linewidth=2, color='red')
    ax4.axhline(y=1.0, linestyle='--', color='gray', label='Target T=1.0')
    ax4.set_xlabel('Step (x10)')
    ax4.set_ylabel('Temperature')
    ax4.set_title('Temperature')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_file, dpi=150)
    print(f"   Saved: {save_file}")
    plt.close()


def main():
    print("=" * 60)
    print("   Lennard-Jones MD Simulation")
    print("   Chapter 21: Molecular Dynamics")
    print("=" * 60)

    # Create simulation
    print("\n1. Setting up simulation:")
    md = LennardJonesMD(
        n_particles=36,
        box_size=10.0,
        temperature=1.0,
        dt=0.005
    )
    print(f"   Particles: {md.n}")
    print(f"   Box size: {md.L}")
    print(f"   Target temperature: {md.T}")
    print(f"   Time step: {md.dt}")

    # Run simulation
    print("\n2. Running simulation:")
    n_steps = 5000
    print(f"   Steps: {n_steps}")

    trajectory, energies = md.run_simulation(n_steps=n_steps, save_every=10)

    # Results
    print("\n3. Results:")
    print(f"   Initial total energy: {energies['total'][0]:.3f}")
    print(f"   Final total energy: {energies['total'][-1]:.3f}")
    print(f"   Energy drift: {abs(energies['total'][-1] - energies['total'][0]):.6f}")
    print(f"   Average temperature: {np.mean(energies['temperature']):.3f}")

    # Plot
    print("\n4. Generating plots...")
    plot_results(trajectory, energies)

    print("\n" + "=" * 60)
    print("   Lennard-Jones simulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
