#!/usr/bin/env python3
"""
Dask Basics - พื้นฐาน Dask
Chapter 7: Dask for Parallel Computing

แสดง: Lazy evaluation, Task graphs, Parallel computation
"""

import time
import dask
import dask.array as da
from dask import delayed


def demonstrate_lazy_evaluation():
    """สาธิต Lazy Evaluation"""
    print("\n1️⃣ Lazy Evaluation:")

    # Create large array (doesn't compute yet)
    x = da.random.random((10000, 10000), chunks=(1000, 1000))
    print(f"   Created array: {x}")
    print(f"   Shape: {x.shape}")
    print(f"   Chunks: {x.chunks}")
    print(f"   Size in memory (if computed): {x.nbytes / 1e9:.2f} GB")

    # Chain operations (still lazy)
    y = x + x.T
    z = y.mean()
    print(f"\n   Operations defined but NOT computed yet")
    print(f"   z = (x + x.T).mean()")

    # Trigger computation
    print(f"\n   Computing...")
    start = time.time()
    result = z.compute()
    elapsed = time.time() - start
    print(f"   Result: {result:.6f}")
    print(f"   Time: {elapsed:.2f}s")


def demonstrate_delayed():
    """สาธิต @delayed decorator"""
    print("\n2️⃣ Delayed Functions:")

    @delayed
    def slow_square(x):
        time.sleep(0.1)  # Simulate slow operation
        return x ** 2

    @delayed
    def slow_sum(values):
        time.sleep(0.1)
        return sum(values)

    # Build task graph
    values = [1, 2, 3, 4, 5]
    squared = [slow_square(v) for v in values]
    total = slow_sum(squared)

    print(f"   Task graph created for: sum([x² for x in {values}])")

    # Sequential time estimate
    sequential_time = len(values) * 0.1 + 0.1
    print(f"   Sequential time estimate: {sequential_time:.1f}s")

    # Parallel execution
    start = time.time()
    result = total.compute()
    elapsed = time.time() - start

    print(f"   Result: {result}")
    print(f"   Parallel time: {elapsed:.2f}s")
    print(f"   Speedup: {sequential_time/elapsed:.1f}x")


def demonstrate_chunks():
    """สาธิตการทำงานกับ chunks"""
    print("\n3️⃣ Chunked Operations:")

    # Create chunked array
    x = da.arange(100, chunks=25)
    print(f"   Array: 0 to 99, chunks of 25")
    print(f"   Chunks: {x.chunks}")

    # Operations preserve chunks
    y = x * 2
    z = y.sum()

    print(f"\n   Operation: (x * 2).sum()")
    print(f"   Result: {z.compute()}")


def demonstrate_memory_efficiency():
    """สาธิตการประมวลผลข้อมูลใหญ่กว่า RAM"""
    print("\n4️⃣ Memory Efficiency:")

    # This would be 800 GB if loaded all at once!
    print("   Creating 100,000 x 100,000 array (80 GB if dense)")
    print("   But with chunks, we process piece by piece")

    x = da.random.random((100000, 100000), chunks=(10000, 10000))

    # Compute mean without loading entire array
    print(f"\n   Computing mean across all 10 billion elements...")
    start = time.time()
    result = x.mean().compute()
    elapsed = time.time() - start

    print(f"   Mean: {result:.6f}")
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Peak memory: much less than 80 GB!")


def main():
    print("=" * 60)
    print("🔷 Dask Basics")
    print("   Chapter 7: Dask for Parallel Computing")
    print("=" * 60)

    demonstrate_lazy_evaluation()
    demonstrate_delayed()
    demonstrate_chunks()

    # Only run memory demo if explicitly requested
    import sys
    if "--full" in sys.argv:
        demonstrate_memory_efficiency()
    else:
        print("\n💡 Run with --full for memory efficiency demo")

    print("\n" + "=" * 60)
    print("✅ Dask basics complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
