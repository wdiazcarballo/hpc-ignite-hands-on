#!/usr/bin/env python3
"""
Crystal Structures - โครงสร้างผลึก
Chapter 23: Materials Science

ศึกษาโครงสร้างผลึกและคุณสมบัติพื้นฐาน
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def create_simple_cubic():
    """สร้างโครงสร้าง Simple Cubic"""
    # Lattice parameter
    a = 1.0

    # Atom positions (corner atoms)
    positions = np.array([
        [0, 0, 0],
        [a, 0, 0],
        [0, a, 0],
        [0, 0, a],
        [a, a, 0],
        [a, 0, a],
        [0, a, a],
        [a, a, a]
    ])

    # Atoms per unit cell (corners shared by 8 cells)
    atoms_per_cell = 8 * (1/8)  # = 1

    return positions, a, atoms_per_cell


def create_bcc():
    """สร้างโครงสร้าง Body-Centered Cubic"""
    a = 1.0

    # Corner atoms + center atom
    positions = np.array([
        [0, 0, 0],
        [a, 0, 0],
        [0, a, 0],
        [0, 0, a],
        [a, a, 0],
        [a, 0, a],
        [0, a, a],
        [a, a, a],
        [a/2, a/2, a/2]  # Body center
    ])

    atoms_per_cell = 8 * (1/8) + 1  # = 2

    return positions, a, atoms_per_cell


def create_fcc():
    """สร้างโครงสร้าง Face-Centered Cubic"""
    a = 1.0

    # Corner atoms + face center atoms
    positions = np.array([
        # Corners
        [0, 0, 0],
        [a, 0, 0],
        [0, a, 0],
        [0, 0, a],
        [a, a, 0],
        [a, 0, a],
        [0, a, a],
        [a, a, a],
        # Face centers
        [a/2, a/2, 0],
        [a/2, 0, a/2],
        [0, a/2, a/2],
        [a/2, a/2, a],
        [a/2, a, a/2],
        [a, a/2, a/2]
    ])

    atoms_per_cell = 8 * (1/8) + 6 * (1/2)  # = 4

    return positions, a, atoms_per_cell


def calculate_packing_fraction(structure_type: str) -> float:
    """คำนวณ Atomic Packing Factor (APF)"""
    if structure_type == 'SC':
        # SC: a = 2r, APF = (1 * 4/3 * pi * r^3) / a^3
        return np.pi / 6  # 0.524

    elif structure_type == 'BCC':
        # BCC: 4r = sqrt(3) * a, APF = (2 * 4/3 * pi * r^3) / a^3
        return np.pi * np.sqrt(3) / 8  # 0.680

    elif structure_type == 'FCC':
        # FCC: 4r = sqrt(2) * a, APF = (4 * 4/3 * pi * r^3) / a^3
        return np.pi / (3 * np.sqrt(2))  # 0.740


def plot_crystal_structures(save_file='crystal_structures.png'):
    """แสดงโครงสร้างผลึกทั้ง 3 แบบ"""
    print("\n2. Plotting Crystal Structures:")

    fig = plt.figure(figsize=(15, 5))

    structures = [
        ('Simple Cubic (SC)', create_simple_cubic()),
        ('Body-Centered Cubic (BCC)', create_bcc()),
        ('Face-Centered Cubic (FCC)', create_fcc())
    ]

    for i, (name, (positions, a, atoms)) in enumerate(structures):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')

        # Plot atoms
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                   s=500, c='blue', alpha=0.6, edgecolors='black')

        # Draw unit cell edges
        for z in [0, a]:
            ax.plot([0, a], [0, 0], [z, z], 'k-', alpha=0.3)
            ax.plot([0, a], [a, a], [z, z], 'k-', alpha=0.3)
            ax.plot([0, 0], [0, a], [z, z], 'k-', alpha=0.3)
            ax.plot([a, a], [0, a], [z, z], 'k-', alpha=0.3)
        for x in [0, a]:
            for y in [0, a]:
                ax.plot([x, x], [y, y], [0, a], 'k-', alpha=0.3)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(f'{name}\nAtoms/cell: {atoms:.0f}')

        ax.set_xlim(-0.2, a + 0.2)
        ax.set_ylim(-0.2, a + 0.2)
        ax.set_zlim(-0.2, a + 0.2)

    plt.tight_layout()
    plt.savefig(save_file, dpi=150)
    print(f"   Saved: {save_file}")
    plt.close()


def compare_real_materials():
    """เปรียบเทียบวัสดุจริง"""
    print("\n3. Real Material Examples:")

    materials = {
        'SC': [('Polonium', 'Po')],
        'BCC': [('Iron (α)', 'Fe'), ('Tungsten', 'W'), ('Chromium', 'Cr')],
        'FCC': [('Copper', 'Cu'), ('Aluminum', 'Al'), ('Gold', 'Au'), ('Silver', 'Ag')]
    }

    print(f"\n   {'Structure':<10} {'Material':<20} {'APF':<8}")
    print("   " + "-" * 40)

    for structure, mat_list in materials.items():
        apf = calculate_packing_fraction(structure)
        for material, symbol in mat_list:
            print(f"   {structure:<10} {material} ({symbol}){'':<10} {apf:.3f}")


def calculate_theoretical_density():
    """คำนวณความหนาแน่นทฤษฎี"""
    print("\n4. Theoretical Density Calculation:")

    # Example: Copper (FCC)
    print("\n   Example: Copper (Cu)")

    A = 63.546  # g/mol (atomic mass)
    a = 3.615e-8  # cm (lattice parameter)
    n = 4  # atoms per unit cell (FCC)
    N_A = 6.022e23  # Avogadro's number

    density = (n * A) / (a**3 * N_A)

    print(f"   Atomic mass: {A} g/mol")
    print(f"   Lattice parameter: {a*1e8:.3f} Å")
    print(f"   Atoms per unit cell: {n}")
    print(f"   Calculated density: {density:.2f} g/cm³")
    print(f"   Experimental density: 8.96 g/cm³")


def demonstrate_miller_indices():
    """สาธิต Miller Indices"""
    print("\n5. Miller Indices:")
    print("   Notation for crystallographic planes and directions")

    planes = {
        '(100)': 'Plane perpendicular to x-axis',
        '(110)': 'Plane cutting x and y axes equally',
        '(111)': 'Plane cutting all axes equally',
        '(200)': 'Plane parallel to (100), half spacing'
    }

    print(f"\n   {'Plane':<10} {'Description':<40}")
    print("   " + "-" * 50)
    for plane, desc in planes.items():
        print(f"   {plane:<10} {desc}")


def main():
    print("=" * 60)
    print("   Crystal Structures")
    print("   Chapter 23: Materials Science")
    print("=" * 60)

    # Basic properties
    print("\n1. Crystal Structure Properties:")
    print(f"\n   {'Structure':<10} {'Atoms/Cell':<15} {'APF':<10} {'CN':<5}")
    print("   " + "-" * 40)

    structures = [
        ('SC', 1, calculate_packing_fraction('SC'), 6),
        ('BCC', 2, calculate_packing_fraction('BCC'), 8),
        ('FCC', 4, calculate_packing_fraction('FCC'), 12)
    ]

    for name, atoms, apf, cn in structures:
        print(f"   {name:<10} {atoms:<15} {apf:<10.3f} {cn:<5}")

    # Plot structures
    plot_crystal_structures()

    # Real materials
    compare_real_materials()

    # Density calculation
    calculate_theoretical_density()

    # Miller indices
    demonstrate_miller_indices()

    print("\n" + "=" * 60)
    print("   Crystal structures complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
