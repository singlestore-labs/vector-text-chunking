#!/usr/bin/env python3
"""Vector similarity search for SingleStore chunks."""

import pymysql
import json
import sys
import os
from sentence_transformers import SentenceTransformer

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

class VectorSearcher:
    def __init__(self):
        """Initialize model and connection."""
        print("Loading embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384

        self.connection = pymysql.connect(
            host=HOST,
            port=PORT,
            user=USER,
            password=PASSWORD,
            database=DATABASE
        )
        print("Connected to SingleStore")

    def vector_search(self, query, top_k=5):
        """Perform vector similarity search."""
        # Generate query embedding
        query_embedding = self.model.encode(query, normalize_embeddings=True)
        query_embedding_json = json.dumps(query_embedding.tolist())

        with self.connection.cursor() as cursor:
            cursor.execute("""
                SELECT
                    chunk_id,
                    text,
                    DOT_PRODUCT(embedding, %s :> VECTOR(%s)) as similarity
                FROM chunks
                WHERE embedding IS NOT NULL
                ORDER BY similarity DESC
                LIMIT %s
            """, (query_embedding_json, self.embedding_dim, top_k))

            return cursor.fetchall()

    def find_similar_chunks(self, chunk_id, top_k=5):
        """Find chunks similar to a given chunk."""
        with self.connection.cursor() as cursor:
            # Get the embedding of the source chunk
            cursor.execute("""
                SELECT embedding
                FROM chunks
                WHERE chunk_id = %s
            """, (chunk_id,))

            result = cursor.fetchone()
            if not result:
                return []

            source_embedding = result[0]

            # Find similar chunks
            cursor.execute("""
                SELECT
                    chunk_id,
                    text,
                    DOT_PRODUCT(embedding, %s) as similarity
                FROM chunks
                WHERE chunk_id != %s AND embedding IS NOT NULL
                ORDER BY similarity DESC
                LIMIT %s
            """, (source_embedding, chunk_id, top_k))

            return cursor.fetchall()

    def search_with_context(self, query, top_k=3, context_window=1):
        """Search and include surrounding chunks for context."""
        results = self.vector_search(query, top_k)

        enriched_results = []
        with self.connection.cursor() as cursor:
            for chunk_id, text, similarity in results:
                # Get surrounding chunks
                cursor.execute("""
                    SELECT chunk_id, SUBSTRING(text, 1, 200) as preview
                    FROM chunks
                    WHERE chunk_id BETWEEN %s AND %s
                    ORDER BY chunk_id
                """, (chunk_id - context_window, chunk_id + context_window))

                context = cursor.fetchall()

                enriched_results.append({
                    'chunk_id': chunk_id,
                    'text': text,
                    'similarity': similarity,
                    'context': context
                })

        return enriched_results

    def close(self):
        self.connection.close()

def main():
    """Main function for command-line usage."""
    if len(sys.argv) < 2:
        print("Usage: python vector_search.py <query>")
        print("       python vector_search.py --similar <chunk_id>")
        print("       python vector_search.py --interactive")
        sys.exit(1)

    searcher = VectorSearcher()

    if sys.argv[1] == '--similar' and len(sys.argv) > 2:
        # Find similar chunks
        chunk_id = int(sys.argv[2])
        print(f"\n🔍 Finding chunks similar to chunk {chunk_id}")
        print("="*60)

        results = searcher.find_similar_chunks(chunk_id, top_k=5)
        for i, (cid, text, similarity) in enumerate(results, 1):
            preview = text[:200].replace('\n', ' ').strip()
            print(f"\n{i}. Chunk {cid} (similarity: {similarity:.4f})")
            print(f"   {preview}...")

    elif sys.argv[1] == '--interactive':
        # Interactive mode
        print("\n📚 Vector Search Interactive Mode")
        print("="*60)
        print("Commands:")
        print("  search <query>    - Vector similarity search")
        print("  similar <id>      - Find similar chunks")
        print("  context <query>   - Search with context")
        print("  quit              - Exit")
        print("-"*60)

        while True:
            try:
                command = input("\n> ").strip()
                if not command:
                    continue

                parts = command.split(maxsplit=1)
                cmd = parts[0].lower()

                if cmd == 'quit':
                    break

                elif cmd == 'search' and len(parts) > 1:
                    query = parts[1]
                    results = searcher.vector_search(query, top_k=5)

                    print(f"\n🔍 Vector search for: '{query}'")
                    for i, (chunk_id, text, similarity) in enumerate(results, 1):
                        preview = text[:200].replace('\n', ' ').strip()
                        print(f"\n{i}. Chunk {chunk_id} (similarity: {similarity:.4f})")
                        print(f"   {preview}...")

                elif cmd == 'similar' and len(parts) > 1:
                    chunk_id = int(parts[1])
                    results = searcher.find_similar_chunks(chunk_id, top_k=3)

                    print(f"\n🔗 Chunks similar to {chunk_id}:")
                    for i, (cid, text, similarity) in enumerate(results, 1):
                        preview = text[:150].replace('\n', ' ').strip()
                        print(f"\n{i}. Chunk {cid} (similarity: {similarity:.4f})")
                        print(f"   {preview}...")

                elif cmd == 'context' and len(parts) > 1:
                    query = parts[1]
                    results = searcher.search_with_context(query, top_k=2)

                    print(f"\n📍 Search with context for: '{query}'")
                    for r in results:
                        print(f"\n🎯 Chunk {r['chunk_id']} (similarity: {r['similarity']:.4f})")
                        preview = r['text'][:200].replace('\n', ' ').strip()
                        print(f"   {preview}...")

                        if r['context']:
                            print("   Context:")
                            for cid, preview in r['context']:
                                marker = ">>>" if cid == r['chunk_id'] else "   "
                                print(f"   {marker} Chunk {cid}: {preview[:100]}...")

                else:
                    print("Unknown command or missing arguments")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

    else:
        # Direct search
        query = ' '.join(sys.argv[1:])
        print(f"\n🔍 Vector search for: '{query}'")
        print("="*60)

        results = searcher.vector_search(query, top_k=5)
        for i, (chunk_id, text, similarity) in enumerate(results, 1):
            preview = text[:300].replace('\n', ' ').strip()
            print(f"\n{i}. Chunk {chunk_id} (similarity: {similarity:.4f})")
            print(f"   {preview}...")

    searcher.close()

if __name__ == "__main__":
    main()