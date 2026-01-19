#!/usr/bin/env python3
"""
GPU Check - ตรวจสอบ GPU และ CUDA
Chapter 4: Deep Learning on HPC

รันก่อน training เพื่อยืนยันว่า GPU พร้อมใช้งาน
"""

import sys


def check_gpu():
    """ตรวจสอบ GPU availability และข้อมูล"""

    print("=" * 60)
    print("🔍 GPU and CUDA Check")
    print("   Chapter 4: Deep Learning on HPC")
    print("=" * 60)

    # Check PyTorch
    try:
        import torch
        print(f"\n✅ PyTorch version: {torch.__version__}")
    except ImportError:
        print("\n❌ PyTorch not installed!")
        sys.exit(1)

    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"\n📊 CUDA Status:")
    print(f"   CUDA available: {cuda_available}")

    if cuda_available:
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   cuDNN version: {torch.backends.cudnn.version()}")

        # GPU count
        gpu_count = torch.cuda.device_count()
        print(f"\n🎮 GPU Count: {gpu_count}")

        # GPU details
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            print(f"\n   GPU {i}: {props.name}")
            print(f"   - Compute capability: {props.major}.{props.minor}")
            print(f"   - Total memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"   - Multi-processor count: {props.multi_processor_count}")

        # Current device
        print(f"\n📍 Current device: {torch.cuda.current_device()}")
        print(f"   Device name: {torch.cuda.get_device_name()}")

        # Memory info
        print(f"\n💾 Memory (GPU 0):")
        print(f"   Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"   Cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")

        # Simple test
        print("\n🧪 Running simple GPU test...")
        try:
            x = torch.randn(1000, 1000, device="cuda")
            y = torch.mm(x, x)
            del x, y
            torch.cuda.empty_cache()
            print("   ✅ GPU computation successful!")
        except Exception as e:
            print(f"   ❌ GPU test failed: {e}")

    else:
        print("\n⚠️ CUDA not available. Running on CPU only.")
        print("   For GPU support on LANTA:")
        print("   module load PyTorch/2.0.1-CUDA-11.7.0")

    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("\n🍎 Apple MPS available (Metal Performance Shaders)")

    print("\n" + "=" * 60)
    print("✅ GPU check complete!")
    print("=" * 60)

    return cuda_available


if __name__ == "__main__":
    check_gpu()
