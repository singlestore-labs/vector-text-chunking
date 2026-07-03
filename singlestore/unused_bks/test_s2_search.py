#!/usr/bin/env python3
"""Test SingleStore search functionality."""

import pymysql
import os
import sys

# Connection details
HOST = 'svc-aac188a1-2ed7-420b-9316-1fc3b9592536-dml.aws-oregon-4.svc.singlestore.com'
PORT = 3306
USER = 'admin'
PASSWORD = os.environ.get('SINGLESTORE_PASSWORD', '')
DATABASE = 'chunks_db'

# Check if password is set
if not PASSWORD:
    print("Error: SINGLESTORE_PASSWORD environment variable not set")
    print("Please set it with: export SINGLESTORE_PASSWORD='your_password'")
    sys.exit(1)

def test_searches():
    connection = pymysql.connect(
        host=HOST,
        port=PORT,
        user=USER,
        password=PASSWORD,
        database=DATABASE
    )

    with connection.cursor() as cursor:
        # First, check what's actually in the table
        print("Sample data from chunks table:")
        cursor.execute("SELECT chunk_id, SUBSTRING(text, 1, 200) FROM chunks LIMIT 3")
        for chunk_id, preview in cursor.fetchall():
            print(f"\nChunk {chunk_id}: {preview}...")

        # Check total count
        cursor.execute("SELECT COUNT(*) FROM chunks")
        total = cursor.fetchone()[0]
        print(f"\nTotal chunks: {total}")

        # Try different search methods
        print("\n" + "="*60)
        print("Testing different search methods:")

        # 1. LIKE search
        print("\n1. LIKE search for 'Elizabeth':")
        cursor.execute("""
            SELECT chunk_id, SUBSTRING(text, 1, 100) as preview
            FROM chunks
            WHERE text LIKE '%Elizabeth%'
            LIMIT 5
        """)
        results = cursor.fetchall()
        print(f"   Found {len(results)} results")
        for chunk_id, preview in results[:2]:
            print(f"   Chunk {chunk_id}: {preview}...")

        # 2. Full-text search with IN BOOLEAN MODE
        print("\n2. Full-text search IN BOOLEAN MODE for 'Elizabeth':")
        cursor.execute("""
            SELECT chunk_id, SUBSTRING(text, 1, 100) as preview
            FROM chunks
            WHERE MATCH(text) AGAINST('Elizabeth' IN BOOLEAN MODE)
            LIMIT 5
        """)
        results = cursor.fetchall()
        print(f"   Found {len(results)} results")
        for chunk_id, preview in results[:2]:
            print(f"   Chunk {chunk_id}: {preview}...")

        # 3. Full-text search with IN NATURAL LANGUAGE MODE
        print("\n3. Full-text search IN NATURAL LANGUAGE MODE for 'Elizabeth':")
        cursor.execute("""
            SELECT chunk_id, SUBSTRING(text, 1, 100) as preview
            FROM chunks
            WHERE MATCH(text) AGAINST('Elizabeth' IN NATURAL LANGUAGE MODE)
            LIMIT 5
        """)
        results = cursor.fetchall()
        print(f"   Found {len(results)} results")
        for chunk_id, preview in results[:2]:
            print(f"   Chunk {chunk_id}: {preview}...")

        # 4. Try with a common word
        print("\n4. Search for common word 'the':")
        cursor.execute("""
            SELECT COUNT(*)
            FROM chunks
            WHERE text LIKE '%the%'
        """)
        count = cursor.fetchone()[0]
        print(f"   LIKE search found: {count} chunks containing 'the'")

        # 5. Check if full-text index exists
        print("\n5. Check indexes on chunks table:")
        cursor.execute("SHOW INDEX FROM chunks")
        indexes = cursor.fetchall()
        for idx in indexes:
            if 'text' in str(idx):
                print(f"   Index: {idx[2]} on column {idx[4]}, Type: {idx[10]}")

        # 6. Rebuild full-text index
        print("\n6. Optimizing table to rebuild indexes...")
        cursor.execute("OPTIMIZE TABLE chunks")
        print("   Table optimized")

        # 7. Try search again after optimize
        print("\n7. Full-text search after optimization for 'Elizabeth':")
        cursor.execute("""
            SELECT chunk_id, SUBSTRING(text, 1, 100) as preview,
                   MATCH(text) AGAINST('Elizabeth') as score
            FROM chunks
            WHERE MATCH(text) AGAINST('Elizabeth')
            ORDER BY score DESC
            LIMIT 5
        """)
        results = cursor.fetchall()
        print(f"   Found {len(results)} results")
        for chunk_id, preview, score in results[:2]:
            print(f"   Chunk {chunk_id} (score: {score}): {preview}...")

    connection.close()

if __name__ == "__main__":
    test_searches()