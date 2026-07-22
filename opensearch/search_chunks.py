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

"""Search chunks in OpenSearch."""

import argparse
from opensearchpy import OpenSearch

def search_chunks(query_text, index_name='book-chunks'):
    # Connect
    from opensearch_config import get_client
    client = get_client()

    # Different query types
    queries = {
        'match': {
            "query": {
                "match": {
                    "text": query_text
                }
            }
        },
        'phrase': {
            "query": {
                "match_phrase": {
                    "text": query_text
                }
            }
        },
        'fuzzy': {
            "query": {
                "fuzzy": {
                    "text": {
                        "value": query_text,
                        "fuzziness": "AUTO"
                    }
                }
            }
        }
    }

    # Run search
    search_body = queries['match'].copy()
    search_body['highlight'] = {
        "fields": {
            "text": {}
        }
    }

    results = client.search(
        index=index_name,
        body=search_body,
        size=5
    )

    print(f"🔍 Found {results['hits']['total']['value']} results for '{query_text}':\n")

    for i, hit in enumerate(results['hits']['hits'], 1):
        print(f"--- Result {i} (Score: {hit['_score']:.2f}) ---")
        print(f"Chunk ID: {hit['_id']}")

        # Show highlight if available
        if 'highlight' in hit:
            print("Highlights:", hit['highlight']['text'][0][:200])
        else:
            print("Text:", hit['_source']['text'][:200] + "...")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('query', help='Search query')
    parser.add_argument('--index', default='book-chunks', help='Index name')
    args = parser.parse_args()

    search_chunks(args.query, args.index)
