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

"""Generate and store embeddings for chunks in SingleStore."""

import pymysql
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import sys
import os
import re
import argparse
from tqdm import tqdm
import torch  # For GPU detection
import pickle  # For checkpointing
import time

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

# Embedding model - using a good open-source model
# MODEL_NAME = 'all-MiniLM-L6-v2'  # 384 dimensions, fast and good quality
# Alternative models:
# 'all-mpnet-base-v2' - 768 dims, higher quality but slower
# 'all-MiniLM-L12-v2' - 384 dims, slightly better than L6
# 'sentence-transformers/all-distilroberta-v1' - 768 dims

# Model configurations for different dimensions
MODEL_CONFIGS = {
    1024: 'BAAI/bge-large-en-v1.5',  # 1024 dimensions
    1536: 'Alibaba-NLP/gte-large-en-v1.5'  # 1536 dimensions
}

def get_model_for_dimension(dim):
    """Get the appropriate model for the requested dimension."""
    if dim not in MODEL_CONFIGS:
        print(f"Error: No model configured for {dim} dimensions")
        print(f"Available dimensions: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)
    return MODEL_CONFIGS[dim]

class EmbeddingGenerator:
    def __init__(self, dimension, checkpoint_file=None):
        """Initialize embedding model and database connection."""
        self.dimension = dimension
        self.column_name = f"embedding_{dimension}"

        # Set checkpoint file name
        self.checkpoint_file = checkpoint_file or f'embedding_checkpoint_{dimension}.pkl'

        # GPU Detection and dynamic batch sizing
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
            self.optimal_batch_size = 256  # Larger batch for GPU
        else:
            print("📊 Using CPU for embeddings")
            self.optimal_batch_size = 128  # Still much larger than 32!

        # Get the appropriate model for this dimension
        model_name = get_model_for_dimension(dimension)
        print(f"Loading embedding model: {model_name} for {dimension} dimensions")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")

        self.connection = pymysql.connect(
            host=HOST,
            port=PORT,
            user=USER,
            password=PASSWORD,
            database=DATABASE,
            autocommit=False
        )

    def setup_vector_column(self):
        """Add vector column to chunks table if it doesn't exist."""
        with self.connection.cursor() as cursor:
            # Check if vector column exists
            cursor.execute("""
                SELECT COLUMN_NAME
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = %s
                AND TABLE_NAME = %s
                AND COLUMN_NAME = %s
            """, (DATABASE, TABLE_NAME, self.column_name))

            if cursor.fetchone():
                print(f"✅ Vector column '{self.column_name}' already exists")
            else:
                # Add vector column
                print(f"Adding vector column '{self.column_name}' (dimension: {self.dimension})...")
                cursor.execute(f"""
                    ALTER TABLE {TABLE_NAME}
                    ADD COLUMN {self.column_name} VECTOR({self.dimension})
                """)
                self.connection.commit()
                print("✅ Vector column added successfully")

            # Skip vector index creation during testing to avoid table locks
            # Vector indexes can be created manually after all data is loaded
            print("ℹ️  Skipping vector index creation (can cause table locks)")

    def load_checkpoint(self):
        """Load progress checkpoint if exists."""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'rb') as f:
                    checkpoint = pickle.load(f)
                print(f"📂 Loaded checkpoint: {len(checkpoint.get('processed_ids', []))} chunks already processed")
                return checkpoint
            except Exception as e:
                print(f"⚠️  Could not load checkpoint: {e}")
        return {'processed_ids': set()}

    def save_checkpoint(self, processed_ids):
        """Save progress checkpoint."""
        try:
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump({'processed_ids': processed_ids}, f)
        except Exception as e:
            print(f"⚠️  Could not save checkpoint: {e}")

    def generate_embeddings(self, batch_size=None, resume=True):
        """Generate embeddings for all chunks - OPTIMIZED VERSION with checkpointing."""
        # Use optimal batch size if not specified
        if batch_size is None:
            batch_size = self.optimal_batch_size

        # Load checkpoint if resuming
        checkpoint = self.load_checkpoint() if resume else {'processed_ids': set()}
        processed_ids = checkpoint.get('processed_ids', set())

        # Track progress timing
        start_time = time.time()

        with self.connection.cursor() as cursor:
            # Get total count
            cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE {self.column_name} IS NULL")
            total = cursor.fetchone()[0]

            if total == 0:
                cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
                total_chunks = cursor.fetchone()[0]
                print(f"ℹ️  All {total_chunks} chunks already have {self.dimension}-dim embeddings")

                response = input("Do you want to regenerate all embeddings? (y/N): ")
                if response.lower() != 'y':
                    return

                # Reset embeddings
                cursor.execute(f"UPDATE {TABLE_NAME} SET {self.column_name} = NULL")
                self.connection.commit()
                total = total_chunks

            print(f"\n📊 Generating {self.dimension}-dim embeddings for {total} chunks...")
            print(f"   Device: {self.device}")
            print(f"   Batch size: {batch_size}")
            print(f"   Model: {get_model_for_dimension(self.dimension)}")

            # Process in batches
            offset = 0
            processed_count = 0
            with tqdm(total=total, desc="Processing chunks") as pbar:
                while offset < total:
                    # Get batch of chunks
                    cursor.execute(f"""
                        SELECT chunk_id, text
                        FROM {TABLE_NAME}
                        WHERE {self.column_name} IS NULL
                        ORDER BY chunk_id
                        LIMIT %s
                    """, (batch_size,))

                    batch = cursor.fetchall()
                    if not batch:
                        break

                    # Filter out already processed chunks if resuming
                    if processed_ids:
                        batch = [(chunk_id, text) for chunk_id, text in batch if chunk_id not in processed_ids]
                        if not batch:
                            offset += batch_size
                            continue

                    chunk_ids = [row[0] for row in batch]
                    texts = [row[1] for row in batch]

                    # Generate embeddings with optimal settings
                    embeddings = self.model.encode(texts,
                                                  batch_size=min(32, batch_size),  # Internal batch size
                                                  show_progress_bar=False,
                                                  convert_to_numpy=True,
                                                  normalize_embeddings=True,  # Normalize for cosine similarity
                                                  device=self.device)

                    # BULK UPDATE - Much faster than individual updates!
                    update_data = []
                    for chunk_id, embedding in zip(chunk_ids, embeddings):
                        # Convert to JSON array format for SingleStore
                        embedding_json = json.dumps(embedding.tolist())
                        update_data.append((embedding_json, chunk_id))

                    # Execute all updates at once using executemany
                    cursor.executemany(f"""
                        UPDATE {TABLE_NAME}
                        SET {self.column_name} = %s
                        WHERE chunk_id = %s
                    """, update_data)

                    self.connection.commit()

                    # Update processed IDs
                    processed_ids.update(chunk_ids)
                    processed_count += len(batch)

                    # Save checkpoint every 1000 chunks
                    if processed_count % 1000 == 0:
                        self.save_checkpoint(processed_ids)

                        # Calculate and display progress stats
                        elapsed = time.time() - start_time
                        rate = processed_count / elapsed if elapsed > 0 else 0
                        remaining = (total - processed_count) / rate if rate > 0 else 0

                        pbar.set_postfix({
                            'rate': f'{rate:.1f}/s',
                            'eta': f'{remaining/60:.1f}min'
                        })

                    pbar.update(len(batch))
                    offset += batch_size

            # Clean up checkpoint file after successful completion
            if os.path.exists(self.checkpoint_file):
                os.remove(self.checkpoint_file)
                print(f"🧹 Cleaned up checkpoint file: {self.checkpoint_file}")

            # Final statistics
            total_time = time.time() - start_time
            print(f"\n✅ Embeddings generated and stored successfully!")
            print(f"   Processed: {processed_count} chunks in {total_time/60:.1f} minutes")
            print(f"   Average rate: {processed_count/total_time:.1f} chunks/second")

    def test_vector_search(self, query_text, top_k=5):
        """Test vector similarity search."""
        print(f"\n🔍 Vector similarity search for: '{query_text}'")
        print("="*60)

        # Generate embedding for query
        query_embedding = self.model.encode(query_text, normalize_embeddings=True)
        query_embedding_json = json.dumps(query_embedding.tolist())

        with self.connection.cursor() as cursor:
            # Vector similarity search using DOT_PRODUCT
            # Since embeddings are normalized, dot product = cosine similarity
            cursor.execute(f"""
                SELECT
                    chunk_id,
                    text,
                    DOT_PRODUCT({self.column_name}, %s :> VECTOR(%s)) as similarity
                FROM {TABLE_NAME}
                WHERE {self.column_name} IS NOT NULL
                ORDER BY similarity DESC
                LIMIT %s
            """, (query_embedding_json, self.dimension, top_k))

            results = cursor.fetchall()

            for i, (chunk_id, text, similarity) in enumerate(results, 1):
                preview = text[:200].replace('\n', ' ').strip()
                print(f"\n{i}. Chunk {chunk_id} (similarity: {similarity:.4f})")
                print(f"   {preview}...")

        return results

    def hybrid_search(self, query_text, top_k=5):
        """Combine vector and full-text search."""
        print(f"\n🎯 Hybrid search for: '{query_text}'")
        print("="*60)

        # Generate embedding for query
        query_embedding = self.model.encode(query_text, normalize_embeddings=True)
        query_embedding_json = json.dumps(query_embedding.tolist())

        with self.connection.cursor() as cursor:
            # Hybrid search - SingleStore doesn't support arithmetic with MATCH...AGAINST directly
            # So we'll use a subquery approach
            cursor.execute(f"""
                SELECT
                    chunk_id,
                    text,
                    vector_score,
                    text_score,
                    (0.7 * vector_score + 0.3 * text_score) as combined_score
                FROM (
                    SELECT
                        chunk_id,
                        text,
                        DOT_PRODUCT({self.column_name}, %s :> VECTOR(%s)) as vector_score,
                        MATCH(text) AGAINST(%s) as text_score
                    FROM {TABLE_NAME}
                    WHERE {self.column_name} IS NOT NULL
                ) as scores
                ORDER BY combined_score DESC
                LIMIT %s
            """, (query_embedding_json, self.dimension, query_text, top_k))

            results = cursor.fetchall()

            for i, (chunk_id, text, vector_score, text_score, combined_score) in enumerate(results, 1):
                preview = text[:200].replace('\n', ' ').strip()
                print(f"\n{i}. Chunk {chunk_id}")
                print(f"   Vector score: {vector_score:.4f}, Text score: {text_score:.4f}, Combined: {combined_score:.4f}")
                print(f"   {preview}...")

        return results

    def close(self):
        """Close database connection."""
        self.connection.close()

def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate embeddings with checkpointing')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start fresh, ignore checkpoint file')
    args = parser.parse_args()

    # Get dimension and batch size from config
    dimension = config.get('embeddings', {}).get('dimension', 1024)
    batch_size = config.get('embeddings', {}).get('batch_size', 128)  # Increased default from 32

    # Validate dimension
    if not isinstance(dimension, int) or dimension <= 0:
        print(f"Error: Invalid dimension '{dimension}'. Must be a positive integer.")
        sys.exit(1)

    print(f"🚀 SingleStore {dimension}-Dimensional Vector Embeddings Generator")
    print("="*60)

    generator = EmbeddingGenerator(dimension)

    try:
        # Setup vector column
        generator.setup_vector_column()

        # Generate embeddings (pass None to use optimal batch size, or specific value from config)
        generator.generate_embeddings(batch_size=batch_size if batch_size else None, resume=not args.no_resume)

        # Test vector search with example queries
        # NOTE: These queries are for Pride & Prejudice sample data.
        # For your own data, expect low similarity scores (<0.5) since these
        # specific terms may not appear in your documents.
        test_queries = [
            "Elizabeth and Darcy's relationship",
            "marriage proposal",
            "pride and prejudice themes",
            "social class and wealth"
        ]

        print("\n" + "="*60)
        print("📋 Testing vector search with sample queries:")
        print("    (These are for Pride & Prejudice demo data)")

        for query in test_queries:
            generator.test_vector_search(query, top_k=3)

        # Test hybrid search (commented out due to SingleStore syntax issues)
        # print("\n" + "="*60)
        # print("📋 Testing hybrid search:")
        # generator.hybrid_search("Elizabeth refuses Darcy", top_k=3)

    finally:
        generator.close()

if __name__ == "__main__":
    main()
