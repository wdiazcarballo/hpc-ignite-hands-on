#!/usr/bin/env python3
"""
NDVI Analysis - การวิเคราะห์ดัชนีพืชพรรณ
Chapter 24: AI for Forest Protection

NDVI = (NIR - Red) / (NIR + Red)
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_sample_bands():
    """สร้างข้อมูลจำลอง Red และ NIR bands"""
    print("\n1. Generating Sample Satellite Bands:")

    np.random.seed(42)
    size = (200, 200)

    # Create terrain with different land cover types
    x, y = np.meshgrid(np.linspace(0, 1, size[1]), np.linspace(0, 1, size[0]))

    # Dense forest (high NIR, low Red)
    forest_mask = ((x - 0.3)**2 + (y - 0.3)**2) < 0.04
    forest_mask |= ((x - 0.7)**2 + (y - 0.6)**2) < 0.03

    # Agricultural land
    agri_mask = (y > 0.6) & (x < 0.5) & ~forest_mask

    # Urban area (low NDVI)
    urban_mask = ((x - 0.8)**2 + (y - 0.2)**2) < 0.02

    # Water body
    water_mask = ((x - 0.5)**2 + (y - 0.5)**2) < 0.01

    # Generate bands
    red = np.ones(size) * 0.3 + np.random.normal(0, 0.02, size)
    nir = np.ones(size) * 0.3 + np.random.normal(0, 0.02, size)

    # Forest: high NIR, low Red
    red[forest_mask] = 0.05 + np.random.normal(0, 0.01, np.sum(forest_mask))
    nir[forest_mask] = 0.50 + np.random.normal(0, 0.05, np.sum(forest_mask))

    # Agriculture: medium NDVI
    red[agri_mask] = 0.10 + np.random.normal(0, 0.02, np.sum(agri_mask))
    nir[agri_mask] = 0.35 + np.random.normal(0, 0.03, np.sum(agri_mask))

    # Urban: low NIR, medium Red
    red[urban_mask] = 0.25 + np.random.normal(0, 0.02, np.sum(urban_mask))
    nir[urban_mask] = 0.30 + np.random.normal(0, 0.02, np.sum(urban_mask))

    # Water: low both, but NIR < Red
    red[water_mask] = 0.10 + np.random.normal(0, 0.01, np.sum(water_mask))
    nir[water_mask] = 0.05 + np.random.normal(0, 0.01, np.sum(water_mask))

    # Clip to valid range
    red = np.clip(red, 0, 1)
    nir = np.clip(nir, 0, 1)

    print(f"   Image size: {size}")
    print(f"   Red band range: [{red.min():.3f}, {red.max():.3f}]")
    print(f"   NIR band range: [{nir.min():.3f}, {nir.max():.3f}]")

    return red, nir


def calculate_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """คำนวณ NDVI"""
    print("\n2. Calculating NDVI:")
    print("   Formula: NDVI = (NIR - Red) / (NIR + Red)")

    # Avoid division by zero
    denominator = nir + red
    denominator[denominator == 0] = 1e-10

    ndvi = (nir - red) / denominator

    print(f"   NDVI range: [{ndvi.min():.3f}, {ndvi.max():.3f}]")
    print(f"   Mean NDVI: {ndvi.mean():.3f}")

    return ndvi


def classify_land_cover(ndvi: np.ndarray) -> np.ndarray:
    """จำแนกประเภทพื้นที่จาก NDVI"""
    print("\n3. Land Cover Classification:")

    # Classification thresholds
    classes = {
        'Water': (-1.0, -0.1),
        'Urban/Bare': (-0.1, 0.1),
        'Sparse Vegetation': (0.1, 0.3),
        'Agriculture': (0.3, 0.5),
        'Dense Forest': (0.5, 1.0)
    }

    land_cover = np.zeros_like(ndvi, dtype=int)

    print(f"\n   {'Class':<20} {'NDVI Range':<15} {'Pixels':<10} {'%':<8}")
    print("   " + "-" * 55)

    total_pixels = ndvi.size
    for i, (name, (low, high)) in enumerate(classes.items()):
        mask = (ndvi >= low) & (ndvi < high)
        land_cover[mask] = i
        count = np.sum(mask)
        pct = count / total_pixels * 100
        print(f"   {name:<20} [{low:>4.1f}, {high:>4.1f}){'':<5} {count:<10} {pct:.1f}%")

    return land_cover


def detect_deforestation(ndvi_before: np.ndarray, ndvi_after: np.ndarray,
                         threshold: float = -0.2) -> np.ndarray:
    """ตรวจจับการเปลี่ยนแปลงพื้นที่ป่า"""
    print("\n4. Deforestation Detection:")

    # Calculate NDVI change
    ndvi_change = ndvi_after - ndvi_before

    # Detect significant decrease
    deforestation = ndvi_change < threshold

    # Also check that area was forest before (high NDVI)
    was_forest = ndvi_before > 0.4
    deforestation = deforestation & was_forest

    affected_pixels = np.sum(deforestation)
    total_forest = np.sum(was_forest)

    print(f"   Change threshold: {threshold}")
    print(f"   Forest pixels (before): {total_forest}")
    print(f"   Deforested pixels: {affected_pixels}")
    print(f"   Deforestation rate: {affected_pixels/total_forest*100:.2f}%")

    return deforestation, ndvi_change


def plot_ndvi_analysis(red, nir, ndvi, land_cover, save_file='ndvi_analysis.png'):
    """สร้างกราฟแสดงผลการวิเคราะห์"""
    print("\n5. Generating Visualization:")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Red band
    im1 = axes[0, 0].imshow(red, cmap='Reds', vmin=0, vmax=0.5)
    axes[0, 0].set_title('Red Band')
    plt.colorbar(im1, ax=axes[0, 0])

    # NIR band
    im2 = axes[0, 1].imshow(nir, cmap='RdYlGn', vmin=0, vmax=0.6)
    axes[0, 1].set_title('NIR Band')
    plt.colorbar(im2, ax=axes[0, 1])

    # NDVI
    im3 = axes[1, 0].imshow(ndvi, cmap='RdYlGn', vmin=-0.5, vmax=0.8)
    axes[1, 0].set_title('NDVI')
    plt.colorbar(im3, ax=axes[1, 0])

    # Land cover classification
    cmap = plt.cm.get_cmap('tab10', 5)
    im4 = axes[1, 1].imshow(land_cover, cmap=cmap, vmin=0, vmax=4)
    axes[1, 1].set_title('Land Cover Classification')
    cbar = plt.colorbar(im4, ax=axes[1, 1], ticks=[0, 1, 2, 3, 4])
    cbar.ax.set_yticklabels(['Water', 'Urban', 'Sparse', 'Agri', 'Forest'])

    for ax in axes.flat:
        ax.axis('off')

    plt.suptitle('NDVI Analysis - Northern Thailand Forest', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_file, dpi=150)
    print(f"   Saved: {save_file}")
    plt.close()


def main():
    print("=" * 60)
    print("   NDVI Analysis for Forest Monitoring")
    print("   Chapter 24: AI for Forest Protection")
    print("=" * 60)

    # Generate sample data
    red, nir = generate_sample_bands()

    # Calculate NDVI
    ndvi = calculate_ndvi(red, nir)

    # Classify land cover
    land_cover = classify_land_cover(ndvi)

    # Simulate deforestation (modify NIR in forest areas)
    print("\n   Simulating 1-year change...")
    np.random.seed(43)
    nir_after = nir.copy()
    deforest_region = ((red < 0.1) & (nir > 0.4) &
                       (np.random.random(nir.shape) < 0.15))
    nir_after[deforest_region] *= 0.3  # Decrease NIR

    ndvi_after = calculate_ndvi(red, nir_after)
    deforestation, ndvi_change = detect_deforestation(ndvi, ndvi_after)

    # Plot results
    plot_ndvi_analysis(red, nir, ndvi, land_cover)

    print("\n" + "=" * 60)
    print("   NDVI analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
