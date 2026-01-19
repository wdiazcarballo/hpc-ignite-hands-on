#!/usr/bin/env python3
"""
Carbon Calculator - คำนวณ Carbon Footprint
Chapter 30: Carbon Verification with HPC

คำนวณปริมาณ CO2 จากการใช้งาน HPC
"""

import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class HardwareProfile:
    """Hardware power consumption profile"""
    name: str
    tdp_watts: float  # Thermal Design Power
    typical_utilization: float = 0.7  # Average utilization

    @property
    def typical_power(self) -> float:
        return self.tdp_watts * self.typical_utilization


# LANTA Hardware Profiles
HARDWARE_PROFILES = {
    'a100': HardwareProfile('NVIDIA A100-SXM4-40GB', tdp_watts=400),
    'a100_80gb': HardwareProfile('NVIDIA A100-SXM4-80GB', tdp_watts=400),
    'cpu_node': HardwareProfile('AMD EPYC 7713 Node', tdp_watts=560),  # 2 x 280W CPUs
}

# Carbon intensity by region (gCO2/kWh)
CARBON_INTENSITY = {
    'thailand': 450,  # Thai grid average
    'thailand_solar': 50,  # Solar power
    'usa_average': 380,
    'europe_average': 280,
    'france': 60,  # Nuclear heavy
    'germany': 350,
}


def calculate_energy_kwh(
    hardware: str,
    duration_hours: float,
    num_units: int = 1,
    pue: float = 1.3
) -> float:
    """
    Calculate energy consumption in kWh

    Args:
        hardware: Hardware type from HARDWARE_PROFILES
        duration_hours: Job duration in hours
        num_units: Number of hardware units (GPUs, nodes)
        pue: Power Usage Effectiveness (datacenter overhead)

    Returns:
        Total energy in kWh
    """
    profile = HARDWARE_PROFILES.get(hardware)
    if not profile:
        raise ValueError(f"Unknown hardware: {hardware}")

    # Calculate raw energy
    power_kw = profile.typical_power / 1000  # Convert to kW
    energy_raw = power_kw * duration_hours * num_units

    # Apply PUE (datacenter overhead for cooling, etc.)
    energy_total = energy_raw * pue

    return energy_total


def calculate_carbon_kg(
    energy_kwh: float,
    region: str = 'thailand'
) -> float:
    """
    Calculate carbon emissions in kg CO2

    Args:
        energy_kwh: Energy consumption in kWh
        region: Region for carbon intensity

    Returns:
        Carbon emissions in kg CO2
    """
    intensity = CARBON_INTENSITY.get(region, CARBON_INTENSITY['thailand'])
    carbon_g = energy_kwh * intensity
    carbon_kg = carbon_g / 1000

    return carbon_kg


def estimate_job_carbon(
    gpu_type: str = 'a100',
    num_gpus: int = 1,
    duration_hours: float = 1.0,
    region: str = 'thailand',
    pue: float = 1.3
) -> dict:
    """
    Estimate carbon footprint for an HPC job

    Returns:
        Dictionary with energy and carbon estimates
    """
    energy_kwh = calculate_energy_kwh(
        hardware=gpu_type,
        duration_hours=duration_hours,
        num_units=num_gpus,
        pue=pue
    )

    carbon_kg = calculate_carbon_kg(energy_kwh, region)

    # Equivalents
    km_car = carbon_kg / 0.21  # Average car: 210g CO2/km
    tree_hours = carbon_kg / 0.022  # Tree absorbs ~22g CO2/hour

    return {
        'energy_kwh': energy_kwh,
        'carbon_kg': carbon_kg,
        'carbon_g': carbon_kg * 1000,
        'equivalent_car_km': km_car,
        'equivalent_tree_hours': tree_hours,
        'hardware': gpu_type,
        'num_gpus': num_gpus,
        'duration_hours': duration_hours,
        'region': region,
        'pue': pue
    }


def print_report(estimate: dict):
    """Print carbon footprint report"""
    print("\n" + "=" * 50)
    print("   Carbon Footprint Report")
    print("=" * 50)

    print(f"\n   Job Configuration:")
    print(f"   - Hardware: {estimate['hardware'].upper()}")
    print(f"   - GPUs: {estimate['num_gpus']}")
    print(f"   - Duration: {estimate['duration_hours']:.1f} hours")
    print(f"   - Region: {estimate['region'].title()}")
    print(f"   - PUE: {estimate['pue']}")

    print(f"\n   Energy Consumption:")
    print(f"   - Total: {estimate['energy_kwh']:.2f} kWh")

    print(f"\n   Carbon Emissions:")
    print(f"   - Total: {estimate['carbon_kg']:.3f} kg CO2")
    print(f"   - Total: {estimate['carbon_g']:.1f} g CO2")

    print(f"\n   Equivalents:")
    print(f"   - Driving: {estimate['equivalent_car_km']:.1f} km by car")
    print(f"   - Trees: {estimate['equivalent_tree_hours']:.1f} hours to offset")

    print("\n" + "=" * 50)


def compare_scenarios():
    """Compare different training scenarios"""
    print("\n   Scenario Comparison:")
    print("   " + "-" * 60)

    scenarios = [
        ("Small model (1 GPU, 1h)", 'a100', 1, 1.0),
        ("Medium training (4 GPUs, 8h)", 'a100', 4, 8.0),
        ("Large training (8 GPUs, 24h)", 'a100', 8, 24.0),
        ("LLM finetuning (16 GPUs, 48h)", 'a100', 16, 48.0),
    ]

    print(f"\n   {'Scenario':<35} {'Energy':>10} {'Carbon':>12}")
    print("   " + "-" * 60)

    for name, gpu, num, hours in scenarios:
        est = estimate_job_carbon(gpu, num, hours)
        print(f"   {name:<35} {est['energy_kwh']:>8.1f} kWh {est['carbon_kg']:>8.2f} kg")


def green_recommendations(estimate: dict):
    """Provide green computing recommendations"""
    print("\n   Green Computing Recommendations:")
    print("   " + "-" * 50)

    recommendations = []

    # Check duration
    if estimate['duration_hours'] > 24:
        recommendations.append("Consider checkpoint/restart to avoid wasted compute")

    # Check GPU count
    if estimate['num_gpus'] > 4:
        recommendations.append("Profile scaling efficiency - more GPUs may not be faster")

    # Check carbon
    if estimate['carbon_kg'] > 10:
        recommendations.append("Consider running during off-peak hours (lower grid intensity)")
        recommendations.append("Use mixed precision training to reduce time/energy")

    # General recommendations
    recommendations.extend([
        "Use efficient data loading to minimize GPU idle time",
        "Early stopping if validation loss plateaus",
        "Consider model distillation for deployment",
    ])

    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")


def main():
    parser = argparse.ArgumentParser(description='HPC Carbon Footprint Calculator')
    parser.add_argument('--gpu-type', default='a100', choices=list(HARDWARE_PROFILES.keys()))
    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--hours', type=float, default=1.0)
    parser.add_argument('--region', default='thailand', choices=list(CARBON_INTENSITY.keys()))
    parser.add_argument('--compare', action='store_true', help='Show scenario comparison')
    args = parser.parse_args()

    print("=" * 60)
    print("   Carbon Footprint Calculator")
    print("   Chapter 30: Carbon Verification with HPC")
    print("=" * 60)

    # Calculate estimate
    estimate = estimate_job_carbon(
        gpu_type=args.gpu_type,
        num_gpus=args.num_gpus,
        duration_hours=args.hours,
        region=args.region
    )

    print_report(estimate)

    if args.compare:
        compare_scenarios()

    green_recommendations(estimate)

    print("\n" + "=" * 60)
    print("   Carbon calculation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
