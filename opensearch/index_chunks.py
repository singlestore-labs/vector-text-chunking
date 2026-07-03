#!/usr/bin/env python3
"""Index text chunks into OpenSearch."""

import json
import argparse
from opensearchpy import OpenSearch

def index_chunks(chunks_file, index_name='book-chunks'):
    # Connect to OpenSearch
    from opensearch_config import get_client
    client = get_client()

    # Load chunks
    with open(chunks_file, 'r') as f:
        data = json.load(f)

    chunks = data.get('chunks', [])
    strategy = data.get('strategy', 'unknown')

    # Index each chunk
    for i, chunk in enumerate(chunks):
        doc = {
            'text': chunk,
            'chunk_id': i,
            'source': 'pride_and_prejudice',
            'strategy': strategy,
            'length': len(chunk)
        }

        response = client.index(
            index=index_name,
            body=doc,
            id=f"{strategy}_{i}"
        )

        if i % 100 == 0:
            print(f"Indexed {i}/{len(chunks)} chunks...")

    print(f"✅ Indexed {len(chunks)} chunks")

    # Refresh index
    client.indices.refresh(index=index_name)

    # Test search
    query = {
        "query": {
            "match": {
                "text": "Elizabeth Bennet"
            }
        }
    }

    results = client.search(index=index_name, body=query, size=3)
    print(f"\n🔍 Test search for 'Elizabeth Bennet' found {results['hits']['total']['value']} results")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('chunks_file', help='JSON file with chunks')
    parser.add_argument('--index', default='book-chunks', help='Index name')
    args = parser.parse_args()

    index_chunks(args.chunks_file, args.index)