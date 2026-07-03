#!/usr/bin/env python3
"""Generate and store embeddings for chunks in SingleStore."""

import pymysql
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import sys
import os
import argparse
from tqdm import tqdm

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
    def __init__(self, dimension):
        """Initialize embedding model and database connection."""
        self.dimension = dimension
        self.column_name = f"embedding_{dimension}"

        # Get the appropriate model for this dimension
        model_name = get_model_for_dimension(dimension)
        print(f"Loading embedding model: {model_name} for {dimension} dimensions")
        self.model = SentenceTransformer(model_name)
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
                AND TABLE_NAME = 'chunks'
                AND COLUMN_NAME = %s
            """, (DATABASE, self.column_name))

            if cursor.fetchone():
                print(f"✅ Vector column '{self.column_name}' already exists")
            else:
                # Add vector column
                print(f"Adding vector column '{self.column_name}' (dimension: {self.dimension})...")
                cursor.execute(f"""
                    ALTER TABLE chunks
                    ADD COLUMN {self.column_name} VECTOR({self.dimension})
                """)
                self.connection.commit()
                print("✅ Vector column added successfully")

            # Add vector index for similarity search
            try:
                cursor.execute("""
                    ALTER TABLE chunks
                    ADD VECTOR INDEX vec_idx (embedding)
                    INDEX_OPTIONS '{
                        "index_type": "IVF_FLAT",
                        "nlist": 256,
                        "metric_type": "DOT_PRODUCT"
                    }'
                """)
                self.connection.commit()
                print("✅ Vector index created")
            except Exception as e:
                if "Duplicate key name" in str(e):
                    print("✅ Vector index already exists")
                else:
                    print(f"⚠️  Could not create vector index: {e}")

    def generate_embeddings(self, batch_size=32):
        """Generate embeddings for all chunks."""
        with self.connection.cursor() as cursor:
            # Get total count
            cursor.execute(f"SELECT COUNT(*) FROM chunks WHERE {self.column_name} IS NULL")
            total = cursor.fetchone()[0]

            if total == 0:
                cursor.execute("SELECT COUNT(*) FROM chunks")
                total_chunks = cursor.fetchone()[0]
                print(f"ℹ️  All {total_chunks} chunks already have {self.dimension}-dim embeddings")

                response = input("Do you want to regenerate all embeddings? (y/N): ")
                if response.lower() != 'y':
                    return

                # Reset embeddings
                cursor.execute(f"UPDATE chunks SET {self.column_name} = NULL")
                self.connection.commit()
                total = total_chunks

            print(f"\n📊 Generating {self.dimension}-dim embeddings for {total} chunks...")
            print(f"   Batch size: {batch_size}")
            print(f"   Model: {get_model_for_dimension(self.dimension)}")

            # Process in batches
            offset = 0
            with tqdm(total=total, desc="Processing chunks") as pbar:
                while offset < total:
                    # Get batch of chunks
                    cursor.execute(f"""
                        SELECT chunk_id, text
                        FROM chunks
                        WHERE {self.column_name} IS NULL
                        ORDER BY chunk_id
                        LIMIT %s
                    """, (batch_size,))

                    batch = cursor.fetchall()
                    if not batch:
                        break

                    chunk_ids = [row[0] for row in batch]
                    texts = [row[1] for row in batch]

                    # Generate embeddings
                    embeddings = self.model.encode(texts,
                                                  show_progress_bar=False,
                                                  convert_to_numpy=True,
                                                  normalize_embeddings=True)  # Normalize for cosine similarity

                    # Update database
                    for chunk_id, embedding in zip(chunk_ids, embeddings):
                        # Convert to JSON array format for SingleStore
                        embedding_json = json.dumps(embedding.tolist())
                        cursor.execute(f"""
                            UPDATE chunks
                            SET {self.column_name} = %s
                            WHERE chunk_id = %s
                        """, (embedding_json, chunk_id))

                    self.connection.commit()
                    pbar.update(len(batch))
                    offset += batch_size

            print("✅ Embeddings generated and stored successfully!")

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
                FROM chunks
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
                    FROM chunks
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
    # Get dimension and batch size from config
    dimension = config.get('embeddings', {}).get('dimension', 1024)
    batch_size = config.get('embeddings', {}).get('batch_size', 32)

    print(f"🚀 SingleStore {dimension}-Dimensional Vector Embeddings Generator")
    print("="*60)

    generator = EmbeddingGenerator(dimension)

    try:
        # Setup vector column
        generator.setup_vector_column()

        # Generate embeddings
        generator.generate_embeddings(batch_size=batch_size)

        # Test vector search
        test_queries = [
            "Elizabeth and Darcy's relationship",
            "marriage proposal",
            "pride and prejudice themes",
            "social class and wealth"
        ]

        print("\n" + "="*60)
        print("📋 Testing vector search with sample queries:")

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