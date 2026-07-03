#!/usr/bin/env python3
"""Optimized embedding generator for large-scale vector processing."""

import pymysql
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import sys
import os
import argparse
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import pickle

# Load configuration
config_file = 'config.json'
if not os.path.exists(config_file):
    print(f"Error: {config_file} not found")
    sys.exit(1)

with open(config_file) as f:
    config = json.load(f)

# Connection details from config
HOST = config['singlestore']['host']
PORT = config['singlestore']['port']
USER = config['singlestore']['user']
PASSWORD = config['singlestore']['password']
DATABASE = config['singlestore']['database']

# Model configurations for different dimensions
MODEL_CONFIGS = {
    1024: 'BAAI/bge-large-en-v1.5',
    1536: 'Alibaba-NLP/gte-large-en-v1.5'
}

class OptimizedEmbeddingGenerator:
    def __init__(self, dimension, batch_size=128, checkpoint_file='embedding_checkpoint.pkl'):
        """Initialize optimized embedding generator.

        Args:
            dimension: Embedding dimension (1024 or 1536)
            batch_size: Batch size for processing (increased from 32)
            checkpoint_file: File to save progress for resumability
        """
        self.dimension = dimension
        self.batch_size = batch_size
        self.checkpoint_file = checkpoint_file
        self.column_name = f"embedding_{dimension}"

        # Check for GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            print(f"🚀 GPU detected! Using {torch.cuda.get_device_name(0)}")
            # Increase batch size for GPU
            self.batch_size = min(256, batch_size * 2)
        else:
            print("📊 Using CPU for embeddings (GPU not available)")

        # Load model
        model_name = MODEL_CONFIGS[dimension]
        print(f"Loading model: {model_name} for {dimension} dimensions")
        self.model = SentenceTransformer(model_name, device=self.device)

        # Connection pool for parallel database operations
        self.connection = None
        self.connect()

    def connect(self):
        """Create database connection."""
        self.connection = pymysql.connect(
            host=HOST,
            port=PORT,
            user=USER,
            password=PASSWORD,
            database=DATABASE,
            autocommit=False
        )

    def setup_vector_column(self):
        """Setup vector column with optimized settings."""
        with self.connection.cursor() as cursor:
            # Check if column exists
            cursor.execute("""
                SELECT COLUMN_NAME
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = %s
                AND TABLE_NAME = 'chunks'
                AND COLUMN_NAME = %s
            """, (DATABASE, self.column_name))

            if not cursor.fetchone():
                print(f"Adding vector column '{self.column_name}'...")
                cursor.execute(f"""
                    ALTER TABLE chunks
                    ADD COLUMN {self.column_name} VECTOR({self.dimension})
                """)
                self.connection.commit()
                print("✅ Vector column added")

    def get_pending_chunks(self, limit=None):
        """Get chunks that need embeddings."""
        with self.connection.cursor() as cursor:
            query = f"""
                SELECT chunk_id, text
                FROM chunks
                WHERE {self.column_name} IS NULL
                ORDER BY chunk_id
            """
            if limit:
                query += f" LIMIT {limit}"

            cursor.execute(query)
            return cursor.fetchall()

    def load_checkpoint(self):
        """Load progress checkpoint if exists."""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'rb') as f:
                return pickle.load(f)
        return {'processed_ids': set()}

    def save_checkpoint(self, checkpoint_data):
        """Save progress checkpoint."""
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)

    def bulk_update_embeddings(self, chunk_embeddings):
        """Bulk update embeddings using multi-value INSERT."""
        if not chunk_embeddings:
            return

        # Create temporary table for bulk update
        with self.connection.cursor() as cursor:
            # Use INSERT ... ON DUPLICATE KEY UPDATE for efficiency
            values = []
            for chunk_id, embedding in chunk_embeddings:
                embedding_json = json.dumps(embedding.tolist())
                values.append(f"({chunk_id}, '{embedding_json}')")

            # Bulk update in batches of 100
            batch_size = 100
            for i in range(0, len(values), batch_size):
                batch = values[i:i+batch_size]
                query = f"""
                    INSERT INTO chunks (chunk_id, {self.column_name})
                    VALUES {','.join(batch)}
                    ON DUPLICATE KEY UPDATE {self.column_name} = VALUES({self.column_name})
                """
                cursor.execute(query)

            self.connection.commit()

    def generate_embeddings_optimized(self, resume=True, max_chunks=None):
        """Generate embeddings with optimizations.

        Args:
            resume: Whether to resume from checkpoint
            max_chunks: Maximum chunks to process (None for all)
        """
        # Load checkpoint if resuming
        checkpoint = self.load_checkpoint() if resume else {'processed_ids': set()}
        processed_ids = checkpoint['processed_ids']

        # Get pending chunks
        pending_chunks = self.get_pending_chunks(max_chunks)

        # Filter out already processed chunks if resuming
        if resume and processed_ids:
            pending_chunks = [(id, text) for id, text in pending_chunks
                             if id not in processed_ids]
            print(f"Resuming from checkpoint: {len(processed_ids)} already processed")

        total_chunks = len(pending_chunks)
        if total_chunks == 0:
            print("No chunks to process")
            return

        print(f"\n📊 Processing {total_chunks} chunks")
        print(f"   Device: {self.device}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Model: {MODEL_CONFIGS[self.dimension]}")

        # Process in larger batches
        start_time = time.time()
        processed_count = 0

        with tqdm(total=total_chunks, desc="Generating embeddings") as pbar:
            for batch_start in range(0, total_chunks, self.batch_size):
                batch_end = min(batch_start + self.batch_size, total_chunks)
                batch = pending_chunks[batch_start:batch_end]

                # Extract texts and IDs
                chunk_ids = [item[0] for item in batch]
                texts = [item[1] for item in batch]

                # Generate embeddings
                embeddings = self.model.encode(
                    texts,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    device=self.device
                )

                # Prepare bulk update data
                chunk_embeddings = list(zip(chunk_ids, embeddings))

                # Bulk update database
                self.bulk_update_embeddings(chunk_embeddings)

                # Update checkpoint
                processed_ids.update(chunk_ids)
                processed_count += len(chunk_ids)

                # Save checkpoint every 1000 chunks
                if processed_count % 1000 == 0:
                    self.save_checkpoint({'processed_ids': processed_ids})

                pbar.update(len(chunk_ids))

                # Calculate and display statistics
                elapsed = time.time() - start_time
                rate = processed_count / elapsed
                remaining = (total_chunks - processed_count) / rate

                if processed_count % 500 == 0:
                    pbar.set_postfix({
                        'rate': f'{rate:.1f}/s',
                        'eta': f'{remaining/60:.1f}min'
                    })

        # Final statistics
        total_time = time.time() - start_time
        print(f"\n✅ Completed {processed_count} embeddings in {total_time/60:.1f} minutes")
        print(f"   Average rate: {processed_count/total_time:.1f} chunks/second")

        # Clean up checkpoint file
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)

    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()

def main():
    parser = argparse.ArgumentParser(description='Optimized embedding generator')
    parser.add_argument('--dimension', '-d', type=int, default=1024,
                       choices=[1024, 1536], help='Embedding dimension')
    parser.add_argument('--batch-size', '-b', type=int, default=128,
                       help='Batch size for processing')
    parser.add_argument('--max-chunks', '-m', type=int,
                       help='Maximum chunks to process')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start fresh, ignore checkpoint')

    args = parser.parse_args()

    print(f"🚀 Optimized SingleStore Vector Embeddings Generator")
    print("="*60)

    generator = OptimizedEmbeddingGenerator(
        dimension=args.dimension,
        batch_size=args.batch_size
    )

    try:
        # Setup vector column
        generator.setup_vector_column()

        # Generate embeddings
        generator.generate_embeddings_optimized(
            resume=not args.no_resume,
            max_chunks=args.max_chunks
        )

    finally:
        generator.close()

if __name__ == "__main__":
    main()