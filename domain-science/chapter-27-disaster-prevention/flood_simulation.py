#!/usr/bin/env python3
"""
Flood Simulation - การจำลองน้ำท่วม
Chapter 27: Disaster Prevention

Simple 2D flood simulation using shallow water equations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class FloodSimulation:
    """Simple 2D flood simulation"""

    def __init__(self, nx: int = 100, ny: int = 100, dx: float = 100.0):
        """
        Initialize simulation

        Args:
            nx, ny: Grid dimensions
            dx: Grid spacing (meters)
        """
        self.nx = nx
        self.ny = ny
        self.dx = dx

        # Physical constants
        self.g = 9.81  # Gravity
        self.manning = 0.03  # Manning's n

        # State variables
        self.h = np.zeros((ny, nx))  # Water depth
        self.u = np.zeros((ny, nx))  # Velocity x
        self.v = np.zeros((ny, nx))  # Velocity y
        self.z = np.zeros((ny, nx))  # Terrain elevation

        # Time step (CFL condition)
        self.dt = 0.5

    def create_terrain(self, terrain_type: str = 'valley'):
        """Create terrain elevation"""
        x = np.linspace(0, self.nx * self.dx, self.nx)
        y = np.linspace(0, self.ny * self.dx, self.ny)
        X, Y = np.meshgrid(x, y)

        if terrain_type == 'valley':
            # V-shaped valley
            center_x = self.nx * self.dx / 2
            self.z = 10 + 0.02 * np.abs(X - center_x)

            # Add hills
            hill1 = 5 * np.exp(-((X - 2000)**2 + (Y - 2000)**2) / 500000)
            hill2 = 8 * np.exp(-((X - 7000)**2 + (Y - 7000)**2) / 600000)
            self.z += hill1 + hill2

        elif terrain_type == 'flat':
            self.z = np.ones((self.ny, self.nx)) * 5
            # Slight slope
            self.z += Y / (self.ny * self.dx) * 2

    def add_water_source(self, x_idx: int, y_idx: int, amount: float):
        """Add water at a specific location"""
        self.h[y_idx, x_idx] += amount

    def add_rainfall(self, intensity: float):
        """Add uniform rainfall (mm/hour)"""
        # Convert mm/hour to m/step
        rain_rate = intensity / 1000 / 3600 * self.dt
        self.h += rain_rate

    def step(self):
        """Advance simulation one time step"""
        # Copy current state
        h_new = self.h.copy()
        u_new = self.u.copy()
        v_new = self.v.copy()

        # Interior points
        for i in range(1, self.ny - 1):
            for j in range(1, self.nx - 1):
                if self.h[i, j] < 0.001:  # Dry cell
                    continue

                # Water surface elevation
                eta = self.z + self.h

                # Gradients
                deta_dx = (eta[i, j+1] - eta[i, j-1]) / (2 * self.dx)
                deta_dy = (eta[i+1, j] - eta[i-1, j]) / (2 * self.dx)

                # Friction (Manning's equation)
                vel = np.sqrt(self.u[i, j]**2 + self.v[i, j]**2)
                if vel > 0 and self.h[i, j] > 0:
                    friction = self.g * self.manning**2 * vel / (self.h[i, j]**(4/3))
                else:
                    friction = 0

                # Update velocities
                u_new[i, j] = self.u[i, j] - self.dt * (self.g * deta_dx + friction * self.u[i, j])
                v_new[i, j] = self.v[i, j] - self.dt * (self.g * deta_dy + friction * self.v[i, j])

                # Update depth (continuity)
                if self.h[i, j] > 0:
                    flux_x = (self.h[i, j+1] * self.u[i, j+1] - self.h[i, j-1] * self.u[i, j-1]) / (2 * self.dx)
                    flux_y = (self.h[i+1, j] * self.v[i+1, j] - self.h[i-1, j] * self.v[i-1, j]) / (2 * self.dx)
                    h_new[i, j] = self.h[i, j] - self.dt * (flux_x + flux_y)

        # Boundary conditions (no flow)
        h_new[0, :] = h_new[1, :]
        h_new[-1, :] = h_new[-2, :]
        h_new[:, 0] = h_new[:, 1]
        h_new[:, -1] = h_new[:, -2]

        u_new[0, :] = 0
        u_new[-1, :] = 0
        u_new[:, 0] = 0
        u_new[:, -1] = 0

        v_new[0, :] = 0
        v_new[-1, :] = 0
        v_new[:, 0] = 0
        v_new[:, -1] = 0

        # Ensure non-negative depth
        h_new = np.maximum(h_new, 0)

        # Update state
        self.h = h_new
        self.u = u_new
        self.v = v_new

    def run(self, n_steps: int, rain_intensity: float = 0,
            source_location: tuple = None, source_rate: float = 0):
        """Run simulation for n_steps"""
        history = {
            'max_depth': [],
            'total_volume': [],
            'flooded_area': []
        }

        for step in range(n_steps):
            # Add water sources
            if rain_intensity > 0:
                self.add_rainfall(rain_intensity)

            if source_location and source_rate > 0:
                self.add_water_source(source_location[0], source_location[1], source_rate)

            # Advance
            self.step()

            # Record statistics
            history['max_depth'].append(self.h.max())
            history['total_volume'].append(self.h.sum() * self.dx**2)
            history['flooded_area'].append((self.h > 0.1).sum() * self.dx**2 / 1e6)  # km²

        return history

    def plot_results(self, save_file: str = 'flood_simulation.png'):
        """Visualize simulation results"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Terrain
        ax1 = axes[0]
        terrain_im = ax1.imshow(self.z, cmap='terrain', origin='lower')
        ax1.set_title('Terrain Elevation (m)')
        plt.colorbar(terrain_im, ax=ax1, label='Elevation (m)')

        # Water depth
        ax2 = axes[1]

        # Custom colormap for water
        colors = ['white', 'lightblue', 'blue', 'darkblue']
        water_cmap = LinearSegmentedColormap.from_list('water', colors)

        # Mask dry areas
        h_masked = np.ma.masked_where(self.h < 0.01, self.h)

        # Plot terrain as background
        ax2.imshow(self.z, cmap='terrain', origin='lower', alpha=0.5)

        # Overlay water
        water_im = ax2.imshow(h_masked, cmap=water_cmap, origin='lower',
                              vmin=0, vmax=max(0.5, self.h.max()))
        ax2.set_title(f'Flood Depth (max: {self.h.max():.2f} m)')
        plt.colorbar(water_im, ax=ax2, label='Water Depth (m)')

        for ax in axes:
            ax.set_xlabel('X (grid cells)')
            ax.set_ylabel('Y (grid cells)')

        plt.tight_layout()
        plt.savefig(save_file, dpi=150)
        print(f"   Saved: {save_file}")
        plt.close()


def main():
    print("=" * 60)
    print("   Flood Simulation")
    print("   Chapter 27: Disaster Prevention")
    print("=" * 60)

    # Create simulation
    print("\n1. Setting up simulation:")
    sim = FloodSimulation(nx=100, ny=100, dx=100)  # 10km x 10km area

    # Create terrain
    sim.create_terrain('valley')
    print(f"   Domain: {sim.nx * sim.dx / 1000:.0f} km x {sim.ny * sim.dx / 1000:.0f} km")
    print(f"   Grid: {sim.nx} x {sim.ny} cells")
    print(f"   Elevation range: {sim.z.min():.1f} - {sim.z.max():.1f} m")

    # Run simulation with heavy rainfall
    print("\n2. Running simulation:")
    print("   Scenario: Heavy monsoon rainfall (50 mm/hour)")

    n_steps = 100
    history = sim.run(
        n_steps=n_steps,
        rain_intensity=50,  # mm/hour
        source_location=(50, 10),  # River source
        source_rate=0.1  # m per step
    )

    # Results
    print("\n3. Results:")
    print(f"   Simulation time: {n_steps * sim.dt:.0f} seconds")
    print(f"   Max water depth: {history['max_depth'][-1]:.2f} m")
    print(f"   Total water volume: {history['total_volume'][-1]/1e6:.2f} million m³")
    print(f"   Flooded area: {history['flooded_area'][-1]:.2f} km²")

    # Flood warning levels
    max_depth = history['max_depth'][-1]
    print("\n4. Flood Warning Level:")
    if max_depth < 0.3:
        print("   Level: NORMAL - No significant flooding")
    elif max_depth < 0.5:
        print("   Level: WATCH - Minor flooding possible")
    elif max_depth < 1.0:
        print("   Level: WARNING - Moderate flooding expected")
    else:
        print("   Level: EMERGENCY - Severe flooding")

    # Plot
    print("\n5. Generating visualization...")
    sim.plot_results()

    print("\n" + "=" * 60)
    print("   Flood simulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
