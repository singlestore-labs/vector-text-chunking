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

"""Vector similarity search for SingleStore chunks.

Usage Examples:
--------------
1. Top-k vector search with dot_product, uses ANN index, returns top 5 most similar chunks:
   python vector_search.py --topk "machine learning" --num_runs 10 

2. Top-k vector search using Euclidean distance (no index used):
   python vector_search.py --topk-noidx "machine learning" --num_runs 10

3. Full table vector search (computes dot product with entire table, returns MAX of dot_products
   (trying to get db to scan the whole table while only returning 1 result to reduce comm overhead)
   python vector_search.py --full "neural networks" --num_runs 10

4. Find similar chunks (returns chunks similar to a given chunk ID):
   python vector_search.py --similar 32411 --num_runs 5

Notes:
- Default number of runs is 5 if --num_runs is not specified
- Timing statistics are shown when multiple runs are performed
- SQL queries are printed before execution for transparency
- --topk uses DOT_PRODUCT with DESC ordering (uses index)
- --topk-noidx uses EUCLIDEAN_DISTANCE with ASC ordering (may not use index)
- --verify option - runs query once and outputs results, use to verify if query results make sense
"""

import pymysql
import json
import sys
import os
import re
from sentence_transformers import SentenceTransformer

# Load configuration
config_file = 'config.json'
if not os.path.exists(config_file):
    print(f"Error: {config_file} not found")
    print("Please create a config.json file with SingleStore connection details")
    sys.exit(1)

try:
    with open(config_file) as f:
        config = json.load(f)
except json.JSONDecodeError as e:
    print(f"Error: Invalid JSON in {config_file}: {e}")
    sys.exit(1)

# Validate required config keys
try:
    _ = config['singlestore']['host']
    _ = config['singlestore']['port']
    _ = config['singlestore']['user']
    _ = config['singlestore']['password']
    _ = config['singlestore']['database']
except KeyError as e:
    print(f"Error: Missing required config key: {e}")
    print("Please check your config.json has all required singlestore fields")
    sys.exit(1)

def validate_identifier(name, identifier_type="identifier"):
    """Validate SQL identifiers to prevent SQL injection."""
    if not re.match(r'^[a-zA-Z0-9_]+$', name):
        print(f"Error: Invalid {identifier_type} '{name}'. Must contain only alphanumeric characters and underscores.")
        sys.exit(1)
    return name

# Connection details from config
HOST = config['singlestore']['host']
PORT = config['singlestore']['port']
USER = config['singlestore']['user']
PASSWORD = config['singlestore']['password']
DATABASE = validate_identifier(config['singlestore']['database'], "database name")
TABLE_NAME = validate_identifier(config['singlestore'].get('table_name', 'chunks'), "table name")

# Get embedding dimension from config
DIMENSION = config.get('embeddings', {}).get('dimension', 1024)

# Validate dimension
if not isinstance(DIMENSION, int) or DIMENSION <= 0:
    print(f"Error: Invalid dimension '{DIMENSION}'. Must be a positive integer.")
    sys.exit(1)

# Model configurations for different dimensions
MODEL_CONFIGS = {
    384: 'all-MiniLM-L6-v2',
    768: 'all-mpnet-base-v2',
    1024: 'BAAI/bge-large-en-v1.5',
    1536: 'Alibaba-NLP/gte-large-en-v1.5'
}



class VectorSearcher:
    def __init__(self):
        """Initialize model and connection."""
        self.dimension = DIMENSION
        self.column_name = f"embedding_{self.dimension}"

        # Get the appropriate model for the dimension
        if self.dimension not in MODEL_CONFIGS:
            print(f"Error: No model configured for {self.dimension} dimensions")
            sys.exit(1)

        model_name = MODEL_CONFIGS[self.dimension]
        print(f"Loading embedding model: {model_name} for {self.dimension} dimensions...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.dimension

        self.connection = pymysql.connect(
            host=HOST,
            port=PORT,
            user=USER,
            password=PASSWORD,
            database=DATABASE
        )
        print("Connected to SingleStore")

    def vector_topk_search(self, query, top_k=5, num_runs=1, verify_mode=False):
        """Perform vector similarity search with timing."""
        import time

        # Generate query embedding once
        query_embedding = self.model.encode(query, normalize_embeddings=True)
        query_embedding_json = json.dumps(query_embedding.tolist())

        if verify_mode:
            sql_query = f"""
                SELECT
                    chunk_id,
                    text,
                    DOT_PRODUCT({self.column_name}, <embedding> :> VECTOR({self.embedding_dim})) as similarity
                FROM {TABLE_NAME}
                ORDER BY similarity DESC
                LIMIT {top_k}
            """
        else:
            sql_query = f"""
                SELECT
                    chunk_id,
                    DOT_PRODUCT({self.column_name}, <embedding> :> VECTOR({self.embedding_dim})) as similarity
                FROM {TABLE_NAME}
                ORDER BY similarity DESC
                LIMIT {top_k}
            """
        print(f"\nExecuting SQL query:\n{sql_query}")

        times = []
        results = None

        for run in range(num_runs):
            start_time = time.time()
            with self.connection.cursor() as cursor:
                if verify_mode:
                    cursor.execute(f"""
                        SELECT
                            chunk_id,
                            text,
                            DOT_PRODUCT({self.column_name}, %s :> VECTOR(%s)) as similarity
                        FROM {TABLE_NAME}
                        ORDER BY similarity DESC
                        LIMIT %s
                    """, (query_embedding_json, self.embedding_dim, top_k))
                else:
                    cursor.execute(f"""
                        SELECT
                            chunk_id,
                            DOT_PRODUCT({self.column_name}, %s :> VECTOR(%s)) as similarity
                        FROM {TABLE_NAME}
                        ORDER BY similarity DESC
                        LIMIT %s
                    """, (query_embedding_json, self.embedding_dim, top_k))
                results = cursor.fetchall()
            query_time = time.time() - start_time
            times.append(query_time)
            print(f"Run {run+1}: {query_time:.3f} seconds")

        return results, times

    def vector_topk_noidx_search(self, query, top_k=5, num_runs=1, verify_mode=False):
        """Perform vector similarity search using Euclidean distance with timing."""
        import time

        # Generate query embedding once
        query_embedding = self.model.encode(query, normalize_embeddings=True)
        query_embedding_json = json.dumps(query_embedding.tolist())

        if verify_mode:
            sql_query = f"""
                SELECT
                    chunk_id,
                    text,
                    EUCLIDEAN_DISTANCE({self.column_name}, <embedding> :> VECTOR({self.embedding_dim})) as distance
                FROM {TABLE_NAME}
                ORDER BY distance ASC
                LIMIT {top_k}
            """
        else:
            sql_query = f"""
                SELECT
                    chunk_id,
                    EUCLIDEAN_DISTANCE({self.column_name}, <embedding> :> VECTOR({self.embedding_dim})) as distance
                FROM {TABLE_NAME}
                ORDER BY distance ASC
                LIMIT {top_k}
            """
        print(f"\nExecuting SQL query:\n{sql_query}")

        times = []
        results = None

        for run in range(num_runs):
            start_time = time.time()
            with self.connection.cursor() as cursor:
                if verify_mode:
                    cursor.execute(f"""
                        SELECT
                            chunk_id,
                            text,
                            EUCLIDEAN_DISTANCE({self.column_name}, %s :> VECTOR(%s)) as distance
                        FROM {TABLE_NAME}
                        ORDER BY distance ASC
                        LIMIT %s
                    """, (query_embedding_json, self.embedding_dim, top_k))
                else:
                    cursor.execute(f"""
                        SELECT
                            chunk_id,
                            EUCLIDEAN_DISTANCE({self.column_name}, %s :> VECTOR(%s)) as distance
                        FROM {TABLE_NAME}
                        ORDER BY distance ASC
                        LIMIT %s
                    """, (query_embedding_json, self.embedding_dim, top_k))
                results = cursor.fetchall()
            query_time = time.time() - start_time
            times.append(query_time)
            print(f"Run {run+1}: {query_time:.3f} seconds")

        return results, times

    def vector_full_search(self, query, num_runs=1):
        """Perform vector similarity search on entire table without ordering with timing."""
        import time

        # Generate query embedding once
        query_embedding = self.model.encode(query, normalize_embeddings=True)
        query_embedding_json = json.dumps(query_embedding.tolist())

        sql_query = f"""
                SELECT
                    MAX(DOT_PRODUCT({self.column_name}, <embedding> :> VECTOR({self.embedding_dim}))) as max_similarity
                FROM {TABLE_NAME}
            """
        print(f"\nExecuting SQL query:\n{sql_query}")

        times = []
        results = None

        for run in range(num_runs):
            start_time = time.time()
            with self.connection.cursor() as cursor:
                cursor.execute(f"""
                    SELECT
                        MAX(DOT_PRODUCT({self.column_name}, %s :> VECTOR(%s))) as max_similarity
                    FROM {TABLE_NAME}
                """, (query_embedding_json, self.embedding_dim))
                results = cursor.fetchall()
            query_time = time.time() - start_time
            times.append(query_time)
            max_similarity = results[0][0] if results and results[0][0] is not None else 0
            print(f"Run {run+1}: {query_time:.3f} seconds (max similarity: {max_similarity:.4f})")

        return results, times

    def find_similar_chunks(self, chunk_id, top_k=5, num_runs=1, verify_mode=False):
        """Find chunks similar to a given chunk with timing."""
        import time

        # First get the source embedding
        with self.connection.cursor() as cursor:
            sql_query_1 = f"""
                SELECT {self.column_name}
                FROM {TABLE_NAME}
                WHERE chunk_id = {chunk_id}
            """
            print(f"\nExecuting SQL query (get embedding):\n{sql_query_1}")

            cursor.execute(f"""
                SELECT {self.column_name}
                FROM {TABLE_NAME}
                WHERE chunk_id = %s
            """, (chunk_id,))

            result = cursor.fetchone()
            if not result:
                return [], []

            source_embedding = result[0]

        # Find similar chunks with timing
        if verify_mode:
            sql_query_2 = f"""
                SELECT
                    chunk_id,
                    text,
                    DOT_PRODUCT({self.column_name}, <source_embedding>) as similarity
                FROM {TABLE_NAME}
                WHERE chunk_id != {chunk_id}
                ORDER BY similarity DESC
                LIMIT {top_k}
            """
        else:
            sql_query_2 = f"""
                SELECT
                    chunk_id,
                    DOT_PRODUCT({self.column_name}, <source_embedding>) as similarity
                FROM {TABLE_NAME}
                WHERE chunk_id != {chunk_id}
                ORDER BY similarity DESC
                LIMIT {top_k}
            """
        print(f"\nExecuting SQL query (find similar):\n{sql_query_2}")

        times = []
        results = None

        for run in range(num_runs):
            start_time = time.time()
            with self.connection.cursor() as cursor:
                if verify_mode:
                    cursor.execute(f"""
                        SELECT
                            chunk_id,
                            text,
                            DOT_PRODUCT({self.column_name}, %s) as similarity
                        FROM {TABLE_NAME}
                        WHERE chunk_id != %s
                        ORDER BY similarity DESC
                        LIMIT %s
                    """, (source_embedding, chunk_id, top_k))
                else:
                    cursor.execute(f"""
                        SELECT
                            chunk_id,
                            DOT_PRODUCT({self.column_name}, %s) as similarity
                        FROM {TABLE_NAME}
                        WHERE chunk_id != %s
                        ORDER BY similarity DESC
                        LIMIT %s
                    """, (source_embedding, chunk_id, top_k))
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
        print("Usage: python vector_search.py --topk <query> [--num_runs <n>] [--verify]")
        print("       python vector_search.py --topk-noidx <query> [--num_runs <n>] [--verify]")
        print("       python vector_search.py --full <query> [--num_runs <n>] [--verify]")
        print("       python vector_search.py --similar <chunk_id> [--num_runs <n>] [--verify]")
        print("\nOptions:")
        print("  --verify     Run once and display results with text (for result verification)")
        print("  --num_runs   Number of times to run the query (default: 5, overridden to 1 with --verify)")
        sys.exit(1)

    # Parse --num_runs if present (default to 5)
    num_runs = 5
    verify_mode = False
    args = sys.argv[1:]

    # Parse --verify flag
    if '--verify' in args:
        verify_mode = True
        num_runs = 1  # Force single run for verification
        args.remove('--verify')

    if '--num_runs' in args:
        num_runs_index = args.index('--num_runs')
        if num_runs_index + 1 < len(args):
            try:
                if not verify_mode:  # Only set num_runs if not in verify mode
                    num_runs = int(args[num_runs_index + 1])
                # Remove --num_runs and its value from args
                args = args[:num_runs_index] + args[num_runs_index + 2:]
            except ValueError:
                print(f"Error: Invalid value for --num_runs: {args[num_runs_index + 1]}")
                sys.exit(1)
        else:
            print("Error: --num_runs requires a number")
            sys.exit(1)

    searcher = VectorSearcher()
    times = []

    if args[0] == '--similar' and len(args) > 1:
        # Find similar chunks
        chunk_id = int(args[1])
        print(f"\n🔍 Finding chunks similar to chunk {chunk_id}")
        if verify_mode:
            print("Verification mode: Running once with text output")
        print(f"Number of runs: {num_runs}")
        print("="*60)

        results, query_times = searcher.find_similar_chunks(chunk_id, top_k=5, num_runs=num_runs, verify_mode=verify_mode)
        times.extend(query_times)

        # Show results if in verify mode
        if verify_mode and results:
            print("\nResults:")
            if len(results[0]) == 3:  # Has text
                for i, (cid, text, similarity) in enumerate(results, 1):
                    preview = text[:300].replace('\n', ' ').strip() if text else ""
                    print(f"\n{i}. Chunk {cid} (similarity: {similarity:.6f})")
                    print(f"   Text: {preview}...")
            else:  # No text (shouldn't happen in verify mode)
                for i, (cid, similarity) in enumerate(results, 1):
                    print(f"{i}. Chunk {cid} (similarity: {similarity:.4f})")

    elif args[0] == '--topk' and len(args) > 1:
        # Vector topk search
        query = ' '.join(args[1:])
        print(f"\n🔍 Vector search for: '{query}'")
        if verify_mode:
            print("Verification mode: Running once with text output")
        print(f"Number of runs: {num_runs}")
        print("="*60)

        results, query_times = searcher.vector_topk_search(query, top_k=5, num_runs=num_runs, verify_mode=verify_mode)
        times.extend(query_times)

        # Show results if in verify mode
        if verify_mode and results:
            print("\nResults:")
            if len(results[0]) == 3:  # Has text
                for i, (chunk_id, text, similarity) in enumerate(results, 1):
                    preview = text[:300].replace('\n', ' ').strip() if text else ""
                    print(f"\n{i}. Chunk {chunk_id} (similarity: {similarity:.6f})")
                    print(f"   Text: {preview}...")
            else:  # No text (shouldn't happen in verify mode)
                for i, (chunk_id, similarity) in enumerate(results, 1):
                    print(f"{i}. Chunk {chunk_id} (similarity: {similarity:.4f})")

    elif args[0] == '--topk-noidx' and len(args) > 1:
        # Vector topk search using Euclidean distance (no index)
        query = ' '.join(args[1:])
        print(f"\n🔍 Vector search (Euclidean distance) for: '{query}'")
        if verify_mode:
            print("Verification mode: Running once with text output")
        print(f"Number of runs: {num_runs}")
        print("="*60)

        results, query_times = searcher.vector_topk_noidx_search(query, top_k=5, num_runs=num_runs, verify_mode=verify_mode)
        times.extend(query_times)

        # Show results if in verify mode
        if verify_mode and results:
            print("\nResults:")
            if len(results[0]) == 3:  # Has text
                for i, (chunk_id, text, distance) in enumerate(results, 1):
                    preview = text[:300].replace('\n', ' ').strip() if text else ""
                    print(f"\n{i}. Chunk {chunk_id} (distance: {distance:.6f})")
                    print(f"   Text: {preview}...")
            else:  # No text (shouldn't happen in verify mode)
                for i, (chunk_id, distance) in enumerate(results, 1):
                    print(f"{i}. Chunk {chunk_id} (distance: {distance:.4f})")

    elif args[0] == '--full' and len(args) > 1:
        # Full table vector search
        query = ' '.join(args[1:])
        print(f"\n🔍 Full table vector search for: '{query}'")
        if verify_mode:
            print("Verification mode: Running once")
        print(f"Number of runs: {num_runs}")
        print("="*60)

        results, query_times = searcher.vector_full_search(query, num_runs=num_runs)
        times.extend(query_times)

        # Show max similarity if in verify mode (always displayed in the function itself)

    else:
        print("Error: Invalid arguments")
        print("Usage: python vector_search.py --topk <query> [--num_runs <n>] [--verify]")
        print("       python vector_search.py --topk-noidx <query> [--num_runs <n>] [--verify]")
        print("       python vector_search.py --full <query> [--num_runs <n>] [--verify]")
        print("       python vector_search.py --similar <chunk_id> [--num_runs <n>] [--verify]")
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
