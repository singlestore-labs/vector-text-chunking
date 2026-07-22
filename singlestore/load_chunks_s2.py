#!/usr/bin/env python3
"""Load chunks into SingleStore cloud."""

import json
import pymysql
import sys
import os
import re
from datetime import datetime

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
test_queries = config.get('test_queries', ['Elizabeth', 'Darcy', 'marriage'])  # Test queries for search

def setup_database():
    """Create database and table."""
    print("Setting up database...")
    connection = pymysql.connect(
        host=HOST,
        port=PORT,
        user=USER,
        password=PASSWORD,
        autocommit=True
    )

    with connection.cursor() as cursor:
        # Create database
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DATABASE}")
        print(f"✅ Database '{DATABASE}' ready")

        # Use database
        cursor.execute(f"USE {DATABASE}")

        # Create chunks table with full-text search
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                chunk_id BIGINT,
                text TEXT,
                source VARCHAR(255),
                strategy VARCHAR(50),
                length INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FULLTEXT KEY idx_text (text)
            )
        """)
        print(f"✅ Table '{TABLE_NAME}' ready")

        # Check if table has data
        cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
        count = cursor.fetchone()[0]
        if count > 0:
            print(f"❌ Error: Table already contains {count} chunks")
            print("   Please manually clear the table if you want to reload.")
            connection.close()
            sys.exit(1)

    connection.close()
    return True

def load_chunks(chunks_file='chunks.json'):
    """Load chunks from JSON file into SingleStore."""
    print(f"\nLoading chunks from {chunks_file}...")

    # Read chunks
    with open(chunks_file, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict) and 'chunks' in data:
        chunks = data['chunks']
    else:
        chunks = data

    print(f"Found {len(chunks)} chunks to load")

    # Connect to database
    connection = pymysql.connect(
        host=HOST,
        port=PORT,
        user=USER,
        password=PASSWORD,
        database=DATABASE,
        autocommit=False
    )

    try:
        with connection.cursor() as cursor:
            # Prepare insert statement
            insert_sql = f"""
                INSERT INTO {TABLE_NAME} (chunk_id, text, source, strategy, length)
                VALUES (%s, %s, %s, %s, %s)
            """

            # Insert chunks in batches
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]

                values = []
                for idx, chunk in enumerate(batch, start=i):
                    # Handle different chunk formats
                    if isinstance(chunk, dict):
                        text = chunk.get('text', chunk.get('content', ''))
                        source = chunk.get('source', 'pride_and_prejudice.txt')
                        strategy = chunk.get('strategy', 'unknown')
                    else:
                        text = str(chunk)
                        source = 'pride_and_prejudice.txt'
                        strategy = 'unknown'

                    values.append((
                        idx,
                        text,
                        source,
                        strategy,
                        len(text)
                    ))

                cursor.executemany(insert_sql, values)
                connection.commit()
                print(f"  Loaded {i+len(batch)}/{len(chunks)} chunks...")

        print(f"✅ Successfully loaded {len(chunks)} chunks")

        # Show statistics
        with connection.cursor() as cursor:
            cursor.execute(f"""
                SELECT
                    strategy,
                    COUNT(*) as count,
                    AVG(length) as avg_length,
                    MIN(length) as min_length,
                    MAX(length) as max_length
                FROM {TABLE_NAME}
                GROUP BY strategy
            """)

            print("\nChunk Statistics:")
            print(f"{'Strategy':<20} {'Count':<10} {'Avg Length':<12} {'Min':<8} {'Max':<8}")
            print("-" * 60)
            for row in cursor.fetchall():
                strategy, count, avg_len, min_len, max_len = row
                print(f"{strategy:<20} {count:<10} {avg_len:<12.0f} {min_len:<8} {max_len:<8}")

    except Exception as e:
        connection.rollback()
        print(f"❌ Error loading chunks: {e}")
        return False
    finally:
        connection.close()

    return True

def test_search():
    """Test full-text search."""
    print("\nTesting search functionality...")

    connection = pymysql.connect(
        host=HOST,
        port=PORT,
        user=USER,
        password=PASSWORD,
        database=DATABASE
    )

    with connection.cursor() as cursor:
        for query in test_queries:
            cursor.execute(f"""
                SELECT
                    chunk_id,
                    SUBSTRING(text, 1, 100) as preview,
                    MATCH(text) AGAINST(%s) as relevance
                FROM {TABLE_NAME}
                WHERE MATCH(text) AGAINST(%s)
                ORDER BY relevance DESC
                LIMIT 3
            """, (query, query))

            results = cursor.fetchall()
            print(f"\n🔍 Search for '{query}': Found {len(results)} results")
            for chunk_id, preview, relevance in results:
                print(f"  Chunk {chunk_id} (relevance: {relevance:.2f}): {preview}...")

    connection.close()

if __name__ == "__main__":
    import sys

    # Get chunks file from command line or use config default
    chunks_file = sys.argv[1] if len(sys.argv) > 1 else config['chunking']['chunks_file']

    # Setup database
    if setup_database():
        # Load chunks
        if load_chunks(chunks_file):
            # Test search (optional - we use vector search instead)
            # test_search()
            print("\n✅ All done! Your chunks are loaded in SingleStore cloud.")
            print(f"   Database: {DATABASE}")
            print(f"   Table: {TABLE_NAME}")
            print(f"\n💡 Next: Run create_embeddings.py to generate vector embeddings")
        else:
            print("❌ Failed to load chunks")
            sys.exit(1)
