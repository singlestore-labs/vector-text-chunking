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
