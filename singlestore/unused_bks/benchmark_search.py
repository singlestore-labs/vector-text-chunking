#!/usr/bin/env python3
"""Benchmark script for vector search performance.

Usage Examples:
--------------
1. Benchmark top-k search:
   python benchmark_search.py --topk "machine learning" 5

2. Benchmark full table search:
   python benchmark_search.py --full "artificial intelligence" 5
"""

import time
import sys
import os
import statistics

# Add parent directory to path to import vector_search
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vector_search import VectorSearcher

def run_benchmark(query, num_runs=5, search_type='topk'):
    """Run vector search multiple times and measure performance."""
    print(f"Running vector search benchmark")
    print(f"Search type: {search_type}")
    print(f"Query: '{query}'")
    print(f"Number of runs: {num_runs}")
    print("=" * 60)

    # Initialize searcher once (includes model loading)
    print("\nInitializing searcher (loading model)...")
    init_start = time.time()
    searcher = VectorSearcher()
    init_time = time.time() - init_start
    print(f"Initialization time: {init_time:.3f} seconds\n")

    times = []
    result_counts = []

    print("Running queries:")
    for i in range(num_runs):
        print(f"  Run {i+1}/{num_runs}...", end=" ")

        # Time the search based on type
        start_time = time.time()
        if search_type == 'topk':
            results = searcher.vector_topk_search(query, top_k=5)
        elif search_type == 'full':
            results = searcher.vector_full_search(query)
        else:
            print(f"\nError: Unknown search type '{search_type}'")
            sys.exit(1)
        end_time = time.time()

        query_time = end_time - start_time
        times.append(query_time)

        if search_type == 'full':
            # Full search returns single max value
            max_similarity = results[0][0] if results and results[0][0] is not None else 0
            print(f"{query_time:.3f} seconds (max similarity: {max_similarity:.4f})")
        else:
            result_counts.append(len(results))
            print(f"{query_time:.3f} seconds ({len(results)} results)")

    # Close connection
    searcher.close()

    # Calculate statistics
    print("\n" + "=" * 60)
    print("TIMING RESULTS:")
    print(f"  Individual times: {[f'{t:.3f}s' for t in times]}")
    print(f"  Average time: {statistics.mean(times):.3f} seconds")
    print(f"  Median time: {statistics.median(times):.3f} seconds")
    print(f"  Min time: {min(times):.3f} seconds")
    print(f"  Max time: {max(times):.3f} seconds")
    if len(times) > 1:
        print(f"  Std deviation: {statistics.stdev(times):.3f} seconds")


    return times

def main():
    """Main function for command-line usage."""
    if len(sys.argv) < 3:
        print("Usage: python benchmark_search.py --topk <query> [num_runs]")
        print("       python benchmark_search.py --full <query> [num_runs]")
        print("\nExamples:")
        print("  python benchmark_search.py --topk 'machine learning' 5")
        print("  python benchmark_search.py --full 'artificial intelligence' 5")
        sys.exit(1)

    search_type = sys.argv[1].replace('--', '')
    if search_type not in ['topk', 'full']:
        print(f"Error: Invalid search type '{sys.argv[1]}'")
        print("Use --topk or --full")
        sys.exit(1)

    query = sys.argv[2]
    num_runs = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    run_benchmark(query, num_runs, search_type)

if __name__ == "__main__":
    main()