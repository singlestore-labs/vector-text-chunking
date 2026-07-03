#!/usr/bin/env python3
"""Fixed SingleStore search functionality."""

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

def search_chunks(query_text):
    """Search for chunks containing query text."""
    connection = pymysql.connect(
        host=HOST,
        port=PORT,
        user=USER,
        password=PASSWORD,
        database=DATABASE
    )

    print(f"\n🔍 Searching for: '{query_text}'")
    print("="*60)

    with connection.cursor() as cursor:
        # Method 1: LIKE search (always works)
        print("\n1. Using LIKE search:")
        cursor.execute("""
            SELECT
                chunk_id,
                SUBSTRING(text, 1, 200) as preview,
                LENGTH(text) as length
            FROM chunks
            WHERE LOWER(text) LIKE LOWER(%s)
            LIMIT 10
        """, (f'%{query_text}%',))

        results = cursor.fetchall()
        print(f"   Found {len(results)} results")

        for i, (chunk_id, preview, length) in enumerate(results[:3], 1):
            # Clean up the preview
            preview = preview.replace('\n', ' ').strip()
            print(f"\n   Result {i}:")
            print(f"   Chunk ID: {chunk_id}, Length: {length}")
            print(f"   Preview: {preview[:150]}...")

        # Method 2: Try MATCH AGAINST (SingleStore syntax)
        print("\n2. Using MATCH AGAINST (full-text):")
        try:
            cursor.execute("""
                SELECT
                    chunk_id,
                    SUBSTRING(text, 1, 200) as preview,
                    MATCH(text) AGAINST(%s) as relevance
                FROM chunks
                WHERE MATCH(text) AGAINST(%s)
                ORDER BY relevance DESC
                LIMIT 10
            """, (query_text, query_text))

            results = cursor.fetchall()
            print(f"   Found {len(results)} results")

            for i, (chunk_id, preview, relevance) in enumerate(results[:3], 1):
                preview = preview.replace('\n', ' ').strip()
                print(f"\n   Result {i}:")
                print(f"   Chunk ID: {chunk_id}, Relevance: {relevance:.2f}")
                print(f"   Preview: {preview[:150]}...")

        except Exception as e:
            print(f"   Full-text search error: {e}")
            print("   Note: Full-text search may require minimum word length or specific index settings")

        # Method 3: Count total occurrences
        print(f"\n3. Statistics for '{query_text}':")
        cursor.execute("""
            SELECT
                COUNT(*) as total_chunks,
                SUM(CASE WHEN LOWER(text) LIKE LOWER(%s) THEN 1 ELSE 0 END) as chunks_with_term
            FROM chunks
        """, (f'%{query_text}%',))

        total, with_term = cursor.fetchone()
        print(f"   Total chunks: {total}")
        print(f"   Chunks containing '{query_text}': {with_term}")
        print(f"   Percentage: {(with_term/total*100):.1f}%")

    connection.close()

def search_multiple_terms():
    """Search for multiple common terms in Pride and Prejudice."""
    terms = ["Elizabeth", "Darcy", "Bennet", "marriage", "love", "pride", "prejudice"]

    connection = pymysql.connect(
        host=HOST,
        port=PORT,
        user=USER,
        password=PASSWORD,
        database=DATABASE
    )

    print("\n📊 Term Frequency Analysis")
    print("="*60)

    with connection.cursor() as cursor:
        for term in terms:
            cursor.execute("""
                SELECT COUNT(*)
                FROM chunks
                WHERE LOWER(text) LIKE LOWER(%s)
            """, (f'%{term}%',))

            count = cursor.fetchone()[0]
            print(f"{term:<15} appears in {count:>4} chunks ({count/9.65:.1f}%)")

    connection.close()

def get_chunk_by_id(chunk_id):
    """Get full text of a specific chunk."""
    connection = pymysql.connect(
        host=HOST,
        port=PORT,
        user=USER,
        password=PASSWORD,
        database=DATABASE
    )

    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT chunk_id, text, length
            FROM chunks
            WHERE chunk_id = %s
        """, (chunk_id,))

        result = cursor.fetchone()
        if result:
            chunk_id, text, length = result
            print(f"\n📄 Chunk {chunk_id} (Length: {length} characters)")
            print("-"*60)
            print(text[:500])
            if length > 500:
                print(f"\n... [{length-500} more characters]")

    connection.close()

if __name__ == "__main__":
    # Search for main characters
    search_chunks("Elizabeth")
    search_chunks("Darcy")

    # Show term frequencies
    search_multiple_terms()

    # Get a specific chunk
    print("\n" + "="*60)
    print("Sample chunk content:")
    get_chunk_by_id(100)