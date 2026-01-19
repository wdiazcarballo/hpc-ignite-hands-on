#!/usr/bin/env python3
"""
Matplotlib Basics - พื้นฐาน Matplotlib
Chapter 6: Data Visualization

แสดง: Line plots, Scatter, Bar charts, Subplots
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_line_chart():
    """สร้าง Line Chart"""
    print("\n1. Line Chart - Amdahl's Law:")

    # Amdahl's Law: Speedup = 1 / ((1-P) + P/N)
    processors = np.arange(1, 65)
    parallel_fractions = [0.5, 0.75, 0.9, 0.95]

    plt.figure(figsize=(10, 6))

    for p in parallel_fractions:
        speedup = 1 / ((1 - p) + p / processors)
        plt.plot(processors, speedup, label=f'P = {p:.0%}', linewidth=2)

    plt.xlabel('Number of Processors', fontsize=12)
    plt.ylabel('Speedup', fontsize=12)
    plt.title("Amdahl's Law: Effect of Parallel Fraction", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(1, 64)

    plt.tight_layout()
    plt.savefig('amdahl_law.png', dpi=150)
    print("   Saved: amdahl_law.png")
    plt.close()


def plot_bar_chart():
    """สร้าง Bar Chart"""
    print("\n2. Bar Chart - HPC Job Statistics:")

    categories = ['Submitted', 'Running', 'Pending', 'Completed', 'Failed']
    values = [1250, 45, 380, 810, 15]
    colors = ['#3498db', '#2ecc71', '#f39c12', '#27ae60', '#e74c3c']

    plt.figure(figsize=(10, 6))

    bars = plt.bar(categories, values, color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                 f'{val:,}', ha='center', va='bottom', fontsize=11)

    plt.xlabel('Job Status', fontsize=12)
    plt.ylabel('Number of Jobs', fontsize=12)
    plt.title('LANTA Job Queue Statistics', fontsize=14)
    plt.ylim(0, max(values) * 1.15)

    plt.tight_layout()
    plt.savefig('job_statistics.png', dpi=150)
    print("   Saved: job_statistics.png")
    plt.close()


def plot_scatter():
    """สร้าง Scatter Plot"""
    print("\n3. Scatter Plot - Speedup vs Problem Size:")

    np.random.seed(42)

    # Simulated data
    problem_sizes = np.logspace(3, 7, 50)  # 1K to 10M
    speedups = 2 + 5 * np.log10(problem_sizes) / 7 + np.random.normal(0, 0.5, 50)
    efficiency = 80 + 15 * np.log10(problem_sizes) / 7 + np.random.normal(0, 5, 50)

    plt.figure(figsize=(10, 6))

    scatter = plt.scatter(problem_sizes, speedups, c=efficiency, cmap='viridis',
                          s=80, alpha=0.7, edgecolors='black', linewidth=0.5)

    plt.colorbar(scatter, label='Parallel Efficiency (%)')
    plt.xscale('log')
    plt.xlabel('Problem Size (elements)', fontsize=12)
    plt.ylabel('Speedup (8 cores)', fontsize=12)
    plt.title('Parallel Speedup vs Problem Size', fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('speedup_scatter.png', dpi=150)
    print("   Saved: speedup_scatter.png")
    plt.close()


def plot_subplots():
    """สร้าง Multiple Subplots"""
    print("\n4. Subplots - HPC Performance Dashboard:")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: CPU Usage over time
    ax1 = axes[0, 0]
    time_hours = np.arange(0, 24, 0.5)
    cpu_usage = 60 + 25 * np.sin(time_hours / 3) + np.random.normal(0, 5, len(time_hours))
    ax1.plot(time_hours, cpu_usage, 'b-', linewidth=2)
    ax1.fill_between(time_hours, cpu_usage, alpha=0.3)
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('CPU Usage (%)')
    ax1.set_title('CPU Utilization')
    ax1.set_xlim(0, 24)
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Memory Usage
    ax2 = axes[0, 1]
    nodes = ['Node 1', 'Node 2', 'Node 3', 'Node 4']
    memory_used = [256, 312, 189, 445]
    memory_total = [512, 512, 512, 512]
    x = np.arange(len(nodes))
    ax2.bar(x, memory_total, color='lightgray', label='Total', width=0.6)
    ax2.bar(x, memory_used, color='#3498db', label='Used', width=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(nodes)
    ax2.set_ylabel('Memory (GB)')
    ax2.set_title('Memory Usage by Node')
    ax2.legend()

    # Plot 3: Job Queue
    ax3 = axes[1, 0]
    status = ['Running', 'Pending', 'Completed']
    counts = [45, 120, 890]
    colors = ['#2ecc71', '#f39c12', '#3498db']
    ax3.pie(counts, labels=status, colors=colors, autopct='%1.1f%%',
            startangle=90, explode=(0.05, 0, 0))
    ax3.set_title('Job Queue Distribution')

    # Plot 4: GPU Utilization
    ax4 = axes[1, 1]
    gpu_ids = [f'GPU {i}' for i in range(4)]
    utilization = [78, 92, 65, 88]
    colors = ['#2ecc71' if u > 70 else '#f39c12' for u in utilization]
    bars = ax4.barh(gpu_ids, utilization, color=colors)
    ax4.set_xlabel('Utilization (%)')
    ax4.set_title('GPU Utilization')
    ax4.set_xlim(0, 100)
    for bar, val in zip(bars, utilization):
        ax4.text(val + 2, bar.get_y() + bar.get_height()/2, f'{val}%', va='center')

    plt.suptitle('LANTA HPC Monitoring Dashboard', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('hpc_dashboard.png', dpi=150, bbox_inches='tight')
    print("   Saved: hpc_dashboard.png")
    plt.close()


def plot_data_ink_comparison():
    """เปรียบเทียบ Data-Ink Ratio"""
    print("\n5. Data-Ink Ratio Comparison:")

    data = [23, 45, 56, 78, 32]
    labels = ['A', 'B', 'C', 'D', 'E']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bad: Low Data-Ink Ratio
    ax1 = axes[0]
    bars1 = ax1.bar(labels, data, color='lightblue', edgecolor='black',
                    linewidth=2, hatch='///')
    ax1.set_facecolor('#f0f0f0')
    ax1.grid(True, which='both', linestyle='--', linewidth=2, color='gray')
    ax1.set_xlabel('Category', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Value', fontsize=14, fontweight='bold')
    ax1.set_title('Low Data-Ink Ratio (Bad)', fontsize=14, fontweight='bold',
                  bbox=dict(boxstyle='round', facecolor='yellow'))
    for bar, val in zip(bars1, data):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f'{val}', ha='center', fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

    # Good: High Data-Ink Ratio
    ax2 = axes[1]
    ax2.bar(labels, data, color='#3498db', width=0.6)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xlabel('Category')
    ax2.set_ylabel('Value')
    ax2.set_title('High Data-Ink Ratio (Good)', fontsize=12)

    plt.tight_layout()
    plt.savefig('data_ink_comparison.png', dpi=150)
    print("   Saved: data_ink_comparison.png")
    plt.close()


def main():
    print("=" * 60)
    print("   Matplotlib Basics")
    print("   Chapter 6: Data Visualization")
    print("=" * 60)

    plot_line_chart()
    plot_bar_chart()
    plot_scatter()
    plot_subplots()
    plot_data_ink_comparison()

    print("\n" + "=" * 60)
    print("   All visualizations generated!")
    print("=" * 60)


if __name__ == "__main__":
    main()
