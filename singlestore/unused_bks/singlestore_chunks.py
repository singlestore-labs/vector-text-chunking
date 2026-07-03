#!/usr/bin/env python3
"""Insert and search text chunks in SingleStore."""

import json
import argparse
import pymysql
from typing import List, Dict

class SingleStoreChunker:
    def __init__(self,
                 host='localhost',
                 port=3306,
                 user='root',
                 password='',
                 database='chunks_db'):
        """Initialize SingleStore connection."""
        self.connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            autocommit=True
        )
        self.database = database
        self._setup_database()

    def _setup_database(self):
        """Create database and tables if they don't exist."""
        with self.connection.cursor() as cursor:
            # Create database
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
            cursor.execute(f"USE {self.database}")

            # Create chunks table with full-text search index
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    chunk_id INT,
                    text TEXT,
                    source VARCHAR(255),
                    strategy VARCHAR(50),
                    length INT,
                    FULLTEXT KEY idx_text (text),
                    KEY idx_source (source),
                    KEY idx_strategy (strategy)
                )
            """)

            print(f"✅ Database '{self.database}' and table 'chunks' ready")

    def insert_chunks(self, chunks_file: str):
        """Insert chunks from JSON file into SingleStore."""
        with open(chunks_file, 'r') as f:
            data = json.load(f)

        chunks = data.get('chunks', [])
        strategy = data.get('strategy', 'unknown')
        source = chunks_file.replace('.json', '')

        with self.connection.cursor() as cursor:
            cursor.execute(f"USE {self.database}")

            # Clear existing chunks from same source/strategy
            cursor.execute(
                "DELETE FROM chunks WHERE source = %s AND strategy = %s",
                (source, strategy)
            )

            # Insert chunks
            insert_query = """
                INSERT INTO chunks (chunk_id, text, source, strategy, length)
                VALUES (%s, %s, %s, %s, %s)
            """

            for i, chunk in enumerate(chunks):
                cursor.execute(insert_query, (
                    i,
                    chunk,
                    source,
                    strategy,
                    len(chunk)
                ))

                if i % 100 == 0:
                    print(f"Inserted {i}/{len(chunks)} chunks...")

            print(f"✅ Inserted {len(chunks)} chunks into SingleStore")

    def search_fulltext(self, query: str, limit: int = 5) -> List[Dict]:
        """Full-text search using SingleStore."""
        with self.connection.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute(f"USE {self.database}")

            # Full-text search with relevance score
            search_query = """
                SELECT
                    id,
                    chunk_id,
                    LEFT(text, 200) as preview,
                    source,
                    strategy,
                    length,
                    MATCH(text) AGAINST(%s) as score
                FROM chunks
                WHERE MATCH(text) AGAINST(%s)
                ORDER BY score DESC
                LIMIT %s
            """

            cursor.execute(search_query, (query, query, limit))
            results = cursor.fetchall()

            return results

    def search_like(self, query: str, limit: int = 5) -> List[Dict]:
        """Simple LIKE search."""
        with self.connection.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute(f"USE {self.database}")

            search_query = """
                SELECT
                    id,
                    chunk_id,
                    LEFT(text, 200) as preview,
                    source,
                    strategy,
                    length
                FROM chunks
                WHERE text LIKE %s
                LIMIT %s
            """

            cursor.execute(search_query, (f'%{query}%', limit))
            results = cursor.fetchall()

            return results

    def get_statistics(self) -> Dict:
        """Get statistics about stored chunks."""
        with self.connection.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute(f"USE {self.database}")

            # Overall stats
            cursor.execute("""
                SELECT
                    COUNT(*) as total_chunks,
                    COUNT(DISTINCT source) as sources,
                    COUNT(DISTINCT strategy) as strategies,
                    AVG(length) as avg_length,
                    MIN(length) as min_length,
                    MAX(length) as max_length
                FROM chunks
            """)
            stats = cursor.fetchone()

            # Per strategy stats
            cursor.execute("""
                SELECT
                    strategy,
                    COUNT(*) as count,
                    AVG(length) as avg_length
                FROM chunks
                GROUP BY strategy
            """)
            by_strategy = cursor.fetchall()

            stats['by_strategy'] = by_strategy
            return stats

    def close(self):
        """Close database connection."""
        self.connection.close()

def main():
    parser = argparse.ArgumentParser(description='SingleStore chunk management')
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Connection args (parent parser to share with all subcommands)
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--host', default='localhost', help='SingleStore host')
    parent_parser.add_argument('--port', type=int, default=3306, help='SingleStore port')
    parent_parser.add_argument('--user', default='root', help='SingleStore user')
    parent_parser.add_argument('--password', default='', help='SingleStore password')
    parent_parser.add_argument('--database', default='chunks_db', help='Database name')

    # Insert command
    insert_parser = subparsers.add_parser('insert', parents=[parent_parser], help='Insert chunks from JSON')
    insert_parser.add_argument('file', help='JSON file with chunks')

    # Search command
    search_parser = subparsers.add_parser('search', parents=[parent_parser], help='Search chunks')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--type', choices=['fulltext', 'like'], default='fulltext')
    search_parser.add_argument('--limit', type=int, default=5)

    # Stats command
    stats_parser = subparsers.add_parser('stats', parents=[parent_parser], help='Show statistics')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Connect to SingleStore
    chunker = SingleStoreChunker(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        database=args.database
    )

    try:
        if args.command == 'insert':
            chunker.insert_chunks(args.file)

        elif args.command == 'search':
            if args.type == 'fulltext':
                results = chunker.search_fulltext(args.query, args.limit)
            else:
                results = chunker.search_like(args.query, args.limit)

            print(f"\n🔍 Found {len(results)} results for '{args.query}':\n")

            for i, result in enumerate(results, 1):
                print(f"--- Result {i} ---")
                print(f"Chunk ID: {result['chunk_id']}")
                print(f"Source: {result['source']}")
                print(f"Strategy: {result['strategy']}")
                if 'score' in result:
                    print(f"Score: {result['score']:.2f}")
                print(f"Preview: {result['preview']}...")
                print()

        elif args.command == 'stats':
            stats = chunker.get_statistics()

            print("\n📊 SingleStore Chunk Statistics:\n")
            print(f"Total chunks: {stats['total_chunks']}")
            print(f"Sources: {stats['sources']}")
            print(f"Strategies: {stats['strategies']}")
            print(f"Avg length: {stats['avg_length']:.0f} chars")
            print(f"Min/Max length: {stats['min_length']}/{stats['max_length']} chars")

            if stats['by_strategy']:
                print("\nBy Strategy:")
                for s in stats['by_strategy']:
                    print(f"  {s['strategy']}: {s['count']} chunks, avg {s['avg_length']:.0f} chars")

    finally:
        chunker.close()

if __name__ == "__main__":
    main()