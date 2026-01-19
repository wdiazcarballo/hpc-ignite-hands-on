#!/usr/bin/env python3
"""
Spark Basics - พื้นฐาน Apache Spark
Chapter 9: Apache Spark for Distributed Computing

แสดง: SparkSession, RDD, Transformations, Actions
"""

import os
import sys

# Check if PySpark is available
try:
    from pyspark.sql import SparkSession
    from pyspark import SparkContext
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False
    print("PySpark not installed. Install with: pip install pyspark")
    print("Running in demo mode (showing concepts only)")


def demonstrate_rdd_basics(sc):
    """สาธิต RDD พื้นฐาน"""
    print("\n1. RDD Basics:")

    # Create RDD from list
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    rdd = sc.parallelize(data, numSlices=4)

    print(f"   Original: {data}")
    print(f"   Number of partitions: {rdd.getNumPartitions()}")

    # Transformations (lazy)
    squared = rdd.map(lambda x: x ** 2)
    evens = squared.filter(lambda x: x % 2 == 0)

    print("   Transformations defined (not computed yet)")

    # Actions (trigger computation)
    result = evens.collect()
    total = evens.reduce(lambda a, b: a + b)

    print(f"   Even squares: {result}")
    print(f"   Sum: {total}")


def demonstrate_word_count(sc):
    """Word Count - ตัวอย่าง classic"""
    print("\n2. Word Count Example:")

    # Sample text
    text = """
    High Performance Computing enables scientific breakthroughs
    LANTA supercomputer provides high performance computing resources
    Thailand researchers use high performance computing for climate modeling
    """

    # Create RDD
    lines = sc.parallelize(text.strip().split('\n'))

    # Word count pipeline
    words = lines.flatMap(lambda line: line.lower().split())
    word_pairs = words.map(lambda word: (word, 1))
    word_counts = word_pairs.reduceByKey(lambda a, b: a + b)

    # Get results
    results = word_counts.collect()

    # Sort and display top words
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    print("   Top 5 words:")
    for word, count in sorted_results[:5]:
        print(f"   '{word}': {count}")


def demonstrate_transformations(sc):
    """สาธิต Transformations หลากหลายประเภท"""
    print("\n3. Common Transformations:")

    data = [1, 2, 3, 4, 5]
    rdd = sc.parallelize(data)

    # map
    mapped = rdd.map(lambda x: x * 2).collect()
    print(f"   map(x * 2): {mapped}")

    # filter
    filtered = rdd.filter(lambda x: x > 2).collect()
    print(f"   filter(x > 2): {filtered}")

    # flatMap
    flat = rdd.flatMap(lambda x: [x, x*10]).collect()
    print(f"   flatMap([x, x*10]): {flat}")

    # distinct
    with_dups = sc.parallelize([1, 2, 2, 3, 3, 3])
    distinct = with_dups.distinct().collect()
    print(f"   distinct([1,2,2,3,3,3]): {distinct}")


def demonstrate_pair_rdd(sc):
    """สาธิต Pair RDD operations"""
    print("\n4. Pair RDD Operations:")

    # Northern Thailand province data
    data = [
        ('Chiang Mai', 450000),
        ('Chiang Rai', 380000),
        ('Chiang Mai', 120000),  # Different crop
        ('Chiang Rai', 95000),
        ('Lamphun', 180000),
    ]

    rdd = sc.parallelize(data)

    # reduceByKey
    totals = rdd.reduceByKey(lambda a, b: a + b).collect()
    print("   reduceByKey (sum by province):")
    for province, total in sorted(totals):
        print(f"      {province}: {total:,} tons")

    # groupByKey
    grouped = rdd.groupByKey().mapValues(list).collect()
    print("\n   groupByKey:")
    for province, values in sorted(grouped):
        print(f"      {province}: {values}")


def demonstrate_actions(sc):
    """สาธิต Actions"""
    print("\n5. Common Actions:")

    rdd = sc.parallelize([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    print(f"   collect(): {rdd.collect()}")
    print(f"   count(): {rdd.count()}")
    print(f"   first(): {rdd.first()}")
    print(f"   take(3): {rdd.take(3)}")
    print(f"   reduce(+): {rdd.reduce(lambda a, b: a + b)}")
    print(f"   sum(): {rdd.sum()}")
    print(f"   mean(): {rdd.mean()}")
    print(f"   min(): {rdd.min()}")
    print(f"   max(): {rdd.max()}")


def run_demo_mode():
    """Run without PySpark (concept demonstration)"""
    print("\n[Demo Mode - PySpark not available]")
    print("\nKey Spark Concepts:")

    print("\n1. RDD (Resilient Distributed Dataset):")
    print("   - Immutable distributed collection")
    print("   - Fault-tolerant through lineage")
    print("   - Lazy evaluation")

    print("\n2. Transformations (Lazy):")
    print("   - map(func): Apply function to each element")
    print("   - filter(func): Keep elements where func returns True")
    print("   - flatMap(func): Map then flatten")
    print("   - reduceByKey(func): Combine values by key")

    print("\n3. Actions (Trigger Execution):")
    print("   - collect(): Return all elements")
    print("   - count(): Return number of elements")
    print("   - reduce(func): Aggregate elements")
    print("   - take(n): Return first n elements")

    print("\n4. Example Pipeline:")
    print("   sc.textFile('data.txt')")
    print("     .flatMap(lambda line: line.split())")
    print("     .map(lambda word: (word, 1))")
    print("     .reduceByKey(lambda a, b: a + b)")
    print("     .collect()")


def main():
    print("=" * 60)
    print("   Spark Basics")
    print("   Chapter 9: Apache Spark for Distributed Computing")
    print("=" * 60)

    if not PYSPARK_AVAILABLE:
        run_demo_mode()
    else:
        # Create Spark session
        spark = SparkSession.builder \
            .appName("SparkBasics") \
            .master("local[*]") \
            .getOrCreate()

        sc = spark.sparkContext
        sc.setLogLevel("WARN")

        print(f"\n   Spark version: {spark.version}")
        print(f"   Master: {sc.master}")

        demonstrate_rdd_basics(sc)
        demonstrate_word_count(sc)
        demonstrate_transformations(sc)
        demonstrate_pair_rdd(sc)
        demonstrate_actions(sc)

        spark.stop()

    print("\n" + "=" * 60)
    print("   Spark basics complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
