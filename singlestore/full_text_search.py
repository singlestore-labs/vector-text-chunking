#!/usr/bin/env python3
# Copyright 2026 SingleStore, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Full-text search for SingleStore chunks.

Usage Examples:
--------------
1. Match search (basic full-text search):
   python full_text_search.py --match "machine learning algorithms"
   python full_text_search.py --match "machine learning" --num_runs 10

2. Boolean search (with AND/OR/NOT operators):
   python full_text_search.py --boolean "machine AND learning NOT deep"
   python full_text_search.py --boolean "neural OR network" --num_runs 10

3. Natural language search:
   python full_text_search.py --natural "what is machine learning"
   python full_text_search.py --natural "how do neural networks work" --num_runs 5

Notes:
- Default number of runs is 5 if --num_runs is not specified
- Timing statistics are shown when multiple runs are performed
- SQL queries are printed before execution for transparency
"""

import pymysql
import json
import sys
import os

# Load configuration
config_file = 'config.json'
if not os.path.exists(config_file):
    print(f"Error: {config_file} not found")
    print("Please create a config.json file with SingleStore connection details")
    sys.exit(1)

with open(config_file) as f:
    config = json.load(f)

# Connection details from config
HOST = config['singlestore']['host']
PORT = config['singlestore']['port']
USER = config['singlestore']['user']
PASSWORD = config['singlestore']['password']
DATABASE = config['singlestore']['database']
TABLE_NAME = config['singlestore']['table_name']  # Required from config


class FullTextSearcher:
    def __init__(self):
        """Initialize database connection."""
        self.connection = pymysql.connect(
            host=HOST,
            port=PORT,
            user=USER,
            password=PASSWORD,
            database=DATABASE
        )
        print("Connected to SingleStore")

    def match_search(self, query, limit=10, num_runs=1):
        """Perform full-text MATCH search with timing."""
        import time

        sql_query = f"""
            SELECT
                chunk_id,
                text,
                MATCH(text) AGAINST(%s) as relevance
            FROM {TABLE_NAME}
            WHERE MATCH(text) AGAINST(%s)
            ORDER BY relevance DESC
            LIMIT {limit}
        """
        print(f"\nExecuting SQL query:\n{sql_query.replace('%s', f"'{query}'")}")

        times = []
        results = None

        for run in range(num_runs):
            start_time = time.time()
            with self.connection.cursor() as cursor:
                cursor.execute(sql_query, (query, query))
                results = cursor.fetchall()
            query_time = time.time() - start_time
            times.append(query_time)
            print(f"Run {run+1}: {query_time:.3f} seconds")

        return results, times

    def boolean_search(self, query, limit=10, num_runs=1):
        """Perform full-text search with boolean operators with timing."""
        import time

        # Boolean mode search
        sql_query = f"""
            SELECT
                chunk_id,
                text,
                MATCH(text) AGAINST(%s IN BOOLEAN MODE) as relevance
            FROM {TABLE_NAME}
            WHERE MATCH(text) AGAINST(%s IN BOOLEAN MODE)
            ORDER BY relevance DESC
            LIMIT {limit}
        """
        print(f"\nExecuting SQL query:\n{sql_query.replace('%s', f"'{query}'")}")

        times = []
        results = None

        for run in range(num_runs):
            start_time = time.time()
            with self.connection.cursor() as cursor:
                cursor.execute(sql_query, (query, query))
                results = cursor.fetchall()
            query_time = time.time() - start_time
            times.append(query_time)
            print(f"Run {run+1}: {query_time:.3f} seconds")

        return results, times

    def natural_language_search(self, query, limit=10, num_runs=1):
        """Perform natural language full-text search with timing."""
        import time

        # Natural language mode search
        sql_query = f"""
            SELECT
                chunk_id,
                text,
                MATCH(text) AGAINST(%s IN NATURAL LANGUAGE MODE) as relevance
            FROM {TABLE_NAME}
            WHERE MATCH(text) AGAINST(%s IN NATURAL LANGUAGE MODE)
            ORDER BY relevance DESC
            LIMIT {limit}
        """
        print(f"\nExecuting SQL query:\n{sql_query.replace('%s', f"'{query}'")}")

        times = []
        results = None

        for run in range(num_runs):
            start_time = time.time()
            with self.connection.cursor() as cursor:
                cursor.execute(sql_query, (query, query))
                results = cursor.fetchall()
            query_time = time.time() - start_time
            times.append(query_time)
            print(f"Run {run+1}: {query_time:.3f} seconds")

        return results, times

    def close(self):
        self.connection.close()


def main():
    """Main function for command-line usage."""
    import time
    import statistics

    if len(sys.argv) < 2:
        print("Usage: python full_text_search.py --match <query> [--num_runs <n>]")
        print("       python full_text_search.py --boolean <query> [--num_runs <n>]")
        print("       python full_text_search.py --natural <query> [--num_runs <n>]")
        sys.exit(1)

    # Parse --num_runs if present (default to 5)
    num_runs = 5
    args = sys.argv[1:]
    if '--num_runs' in args:
        num_runs_index = args.index('--num_runs')
        if num_runs_index + 1 < len(args):
            try:
                num_runs = int(args[num_runs_index + 1])
                # Remove --num_runs and its value from args
                args = args[:num_runs_index] + args[num_runs_index + 2:]
            except ValueError:
                print(f"Error: Invalid value for --num_runs: {args[num_runs_index + 1]}")
                sys.exit(1)
        else:
            print("Error: --num_runs requires a number")
            sys.exit(1)

    searcher = FullTextSearcher()
    times = []

    if args[0] == '--match' and len(args) > 1:
        # Basic MATCH search
        query = ' '.join(args[1:])
        print(f"\n🔍 Full-text MATCH search for: '{query}'")
        print(f"Number of runs: {num_runs}")
        print("="*60)

        results, query_times = searcher.match_search(query, limit=10, num_runs=num_runs)
        times.extend(query_times)

        # Show results from last run
        if num_runs == 1 and results:
            print("\nTop results:")
            for i, (chunk_id, text, relevance) in enumerate(results[:5], 1):
                preview = text[:200].replace('\n', ' ').strip()
                print(f"\n{i}. Chunk {chunk_id} (relevance: {relevance:.4f})")
                print(f"   {preview}...")

    elif args[0] == '--boolean' and len(args) > 1:
        # Boolean mode search
        query = ' '.join(args[1:])
        print(f"\n🔍 Boolean mode search for: '{query}'")
        print(f"Number of runs: {num_runs}")
        print("="*60)

        results, query_times = searcher.boolean_search(query, limit=10, num_runs=num_runs)
        times.extend(query_times)

        # Show results from last run
        if num_runs == 1 and results:
            print("\nTop results:")
            for i, (chunk_id, text, relevance) in enumerate(results[:5], 1):
                preview = text[:200].replace('\n', ' ').strip()
                print(f"\n{i}. Chunk {chunk_id} (relevance: {relevance:.4f})")
                print(f"   {preview}...")

    elif args[0] == '--natural' and len(args) > 1:
        # Natural language search
        query = ' '.join(args[1:])
        print(f"\n🔍 Natural language search for: '{query}'")
        print(f"Number of runs: {num_runs}")
        print("="*60)

        results, query_times = searcher.natural_language_search(query, limit=10, num_runs=num_runs)
        times.extend(query_times)

        # Show results from last run
        if num_runs == 1 and results:
            print("\nTop results:")
            for i, (chunk_id, text, relevance) in enumerate(results[:5], 1):
                preview = text[:200].replace('\n', ' ').strip()
                print(f"\n{i}. Chunk {chunk_id} (relevance: {relevance:.4f})")
                print(f"   {preview}...")

    else:
        print("Error: Invalid arguments")
        print("Usage: python full_text_search.py --match <query> [--num_runs <n>]")
        print("       python full_text_search.py --boolean <query> [--num_runs <n>]")
        print("       python full_text_search.py --natural <query> [--num_runs <n>]")
        sys.exit(1)

    # Show timing statistics if multiple runs
    if num_runs > 1:
        print("\n" + "="*60)
        print("TIMING STATISTICS:")
        print(f"  Average time: {statistics.mean(times):.3f} seconds")
        print(f"  Median time: {statistics.median(times):.3f} seconds")
        print(f"  Min time: {min(times):.3f} seconds")
        print(f"  Max time: {max(times):.3f} seconds")
        if len(times) > 1:
            print(f"  Std deviation: {statistics.stdev(times):.3f} seconds")

    searcher.close()


if __name__ == "__main__":
    main()
