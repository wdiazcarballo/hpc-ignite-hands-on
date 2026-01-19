#!/usr/bin/env python3
"""
Container Demo - สาธิตการใช้ Container
Chapter 11: Containers for HPC

ตัวอย่างการรันโค้ดใน Singularity container
"""

import os
import platform
import sys


def print_environment():
    """แสดงข้อมูล environment"""
    print("\n1. Python Environment:")
    print(f"   Python version: {platform.python_version()}")
    print(f"   Python path: {sys.executable}")
    print(f"   Platform: {platform.platform()}")

    # Check if running in container
    print("\n2. Container Detection:")
    in_container = os.path.exists('/.singularity.d')
    print(f"   Running in Singularity: {in_container}")

    if in_container:
        # Singularity environment variables
        singularity_vars = ['SINGULARITY_CONTAINER', 'SINGULARITY_NAME']
        for var in singularity_vars:
            value = os.environ.get(var, 'Not set')
            print(f"   {var}: {value}")


def check_gpu():
    """ตรวจสอบ GPU"""
    print("\n3. GPU Check:")

    try:
        import torch
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU count: {torch.cuda.device_count()}")
            print(f"   GPU name: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("   PyTorch not installed")

    # NVIDIA driver
    nvidia_driver = os.environ.get('NVIDIA_DRIVER_CAPABILITIES', 'Not set')
    print(f"   NVIDIA_DRIVER_CAPABILITIES: {nvidia_driver}")


def check_libraries():
    """ตรวจสอบ libraries ที่ติดตั้ง"""
    print("\n4. Installed Libraries:")

    libraries = [
        ('numpy', 'np'),
        ('pandas', 'pd'),
        ('scipy', 'scipy'),
        ('matplotlib', 'plt'),
        ('torch', 'torch'),
        ('tensorflow', 'tf'),
    ]

    for name, alias in libraries:
        try:
            module = __import__(name)
            version = getattr(module, '__version__', 'unknown')
            print(f"   {name}: {version}")
        except ImportError:
            print(f"   {name}: not installed")


def run_computation():
    """รันการคำนวณตัวอย่าง"""
    print("\n5. Sample Computation:")

    try:
        import numpy as np

        # Matrix multiplication
        size = 1000
        a = np.random.random((size, size))
        b = np.random.random((size, size))

        import time
        start = time.time()
        c = np.dot(a, b)
        elapsed = time.time() - start

        print(f"   Matrix multiplication ({size}x{size})")
        print(f"   Time: {elapsed:.3f}s")
        print(f"   Result shape: {c.shape}")

    except ImportError:
        print("   NumPy not available")


def main():
    print("=" * 60)
    print("   Container Demo")
    print("   Chapter 11: Containers for HPC")
    print("=" * 60)

    print_environment()
    check_gpu()
    check_libraries()
    run_computation()

    print("\n" + "=" * 60)
    print("   Container demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
