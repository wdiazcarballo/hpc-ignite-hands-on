#!/usr/bin/env python3
"""
PyTorch Basics - พื้นฐาน PyTorch Tensor Operations
Chapter 4: Deep Learning on HPC

ตัวอย่างการใช้งาน PyTorch tensor บน CPU และ GPU
"""

import time
import torch


def tensor_operations():
    """สาธิต tensor operations พื้นฐาน"""

    print("=" * 60)
    print("🔢 PyTorch Tensor Operations")
    print("   Chapter 4: Deep Learning on HPC")
    print("=" * 60)

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n📍 Using device: {device}")

    # 1. Creating tensors
    print("\n1️⃣ Creating Tensors:")
    x = torch.tensor([1, 2, 3, 4])
    print(f"   From list: {x}")

    y = torch.zeros(3, 3)
    print(f"   Zeros (3x3):\n{y}")

    z = torch.randn(2, 3)
    print(f"   Random normal (2x3):\n{z}")

    # 2. Tensor attributes
    print("\n2️⃣ Tensor Attributes:")
    t = torch.randn(3, 4, 5)
    print(f"   Shape: {t.shape}")
    print(f"   Dtype: {t.dtype}")
    print(f"   Device: {t.device}")

    # 3. Moving to GPU
    print("\n3️⃣ Moving to Device:")
    t_device = t.to(device)
    print(f"   After .to({device}): {t_device.device}")

    # 4. Basic operations
    print("\n4️⃣ Basic Operations:")
    a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, device=device)
    b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32, device=device)

    print(f"   a + b =\n{a + b}")
    print(f"   a * b (element-wise) =\n{a * b}")
    print(f"   a @ b (matrix multiply) =\n{a @ b}")

    # 5. Autograd
    print("\n5️⃣ Automatic Differentiation (Autograd):")
    x = torch.tensor([2.0], requires_grad=True, device=device)
    y = x**2 + 3*x + 1  # y = x² + 3x + 1
    y.backward()
    print(f"   f(x) = x² + 3x + 1")
    print(f"   f'(x) = 2x + 3")
    print(f"   f'(2) = {x.grad.item()} (expected: 7)")

    # 6. Performance comparison
    print("\n6️⃣ Performance Comparison (Matrix Multiplication):")
    sizes = [1000, 2000, 4000]

    for size in sizes:
        # CPU
        a_cpu = torch.randn(size, size)
        b_cpu = torch.randn(size, size)

        start = time.time()
        _ = torch.mm(a_cpu, b_cpu)
        cpu_time = time.time() - start

        if device == "cuda":
            # GPU
            a_gpu = a_cpu.to(device)
            b_gpu = b_cpu.to(device)

            # Warm up
            _ = torch.mm(a_gpu, b_gpu)
            torch.cuda.synchronize()

            start = time.time()
            _ = torch.mm(a_gpu, b_gpu)
            torch.cuda.synchronize()
            gpu_time = time.time() - start

            speedup = cpu_time / gpu_time
            print(f"   {size}x{size}: CPU={cpu_time:.4f}s, GPU={gpu_time:.4f}s, Speedup={speedup:.1f}x")
        else:
            print(f"   {size}x{size}: CPU={cpu_time:.4f}s")

    print("\n" + "=" * 60)
    print("✅ PyTorch basics complete!")
    print("=" * 60)


if __name__ == "__main__":
    tensor_operations()
