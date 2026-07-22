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

"""JSON chunker for Wikipedia data (NDJSON format)."""

import json
import argparse
import sys
from tqdm import tqdm
from pathlib import Path

class JSONChunker:
    def __init__(self, max_chunk_size=1000, preserve_metadata=True):
        """Initialize JSON chunker.

        Args:
            max_chunk_size: Maximum characters per chunk
            preserve_metadata: Whether to keep title/url metadata
        """
        self.max_chunk_size = max_chunk_size
        self.preserve_metadata = preserve_metadata

    def chunk_text(self, text, max_size):
        """Split text into chunks at sentence boundaries."""
        if len(text) <= max_size:
            return [text]

        chunks = []
        current_chunk = ""
        sentences = text.replace(". ", ".<<SPLIT>>").split("<<SPLIT>>")

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_size:
                if current_chunk:
                    current_chunk += " "
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def process_article(self, article_data):
        """Process a single Wikipedia article."""
        chunks = []

        # Extract data
        title = article_data.get('title', 'Unknown')
        url = article_data.get('url', '')
        abstract = article_data.get('abstract', '')

        # Skip empty abstracts
        if not abstract or abstract.strip() == '':
            return chunks

        # Chunk the abstract if needed
        text_chunks = self.chunk_text(abstract, self.max_chunk_size)

        for i, chunk_text in enumerate(text_chunks):
            chunk = {
                'text': chunk_text,
                'metadata': {
                    'source': url if url else f"Wikipedia: {title}",
                    'title': title,
                    'chunk_index': i,
                    'total_chunks': len(text_chunks)
                }
            }

            if self.preserve_metadata:
                chunk['metadata']['url'] = url

            chunks.append(chunk)

        return chunks

    def chunk_file(self, filepath, limit=None):
        """Process NDJSON file and return chunks.

        Args:
            filepath: Path to NDJSON file
            limit: Maximum number of articles to process (None for all)
        """
        chunks = []
        articles_processed = 0

        # Count total lines for progress bar (if limit is set)
        total_lines = limit if limit else None
        if not limit:
            print("Counting articles...")
            with open(filepath, 'r', encoding='utf-8') as f:
                total_lines = sum(1 for _ in f)

        print(f"Processing {limit if limit else 'all'} articles from {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            with tqdm(total=total_lines, desc="Processing articles") as pbar:
                for line_num, line in enumerate(f, 1):
                    if limit and articles_processed >= limit:
                        break

                    try:
                        # Parse JSON line
                        article = json.loads(line.strip())

                        # Process article
                        article_chunks = self.process_article(article)
                        chunks.extend(article_chunks)

                        articles_processed += 1
                        pbar.update(1)

                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_num}: {e}")
                        continue
                    except Exception as e:
                        print(f"Error processing line {line_num}: {e}")
                        continue

        print(f"\nProcessed {articles_processed} articles into {len(chunks)} chunks")
        return chunks

def main():
    parser = argparse.ArgumentParser(description='Chunk Wikipedia JSON data')
    parser.add_argument('input_file', help='Path to NDJSON file')
    parser.add_argument('--output', '-o', default='data/wiki_chunks.json',
                       help='Output JSON file (default: data/wiki_chunks.json)')
    parser.add_argument('--size', '-s', type=int, default=1000,
                       help='Maximum chunk size in characters (default: 1000)')
    parser.add_argument('--limit', '-l', type=int,
                       help='Maximum number of articles to process (default: all)')
    parser.add_argument('--sample', action='store_true',
                       help='Process a sample of 1000 articles for testing')

    args = parser.parse_args()

    # Handle sample mode
    if args.sample:
        args.limit = 1000
        print("Sample mode: Processing first 1000 articles")

    # Create chunker
    chunker = JSONChunker(max_chunk_size=args.size)

    # Process file
    chunks = chunker.chunk_file(args.input_file, limit=args.limit)

    # Save chunks
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(chunks)} chunks to {output_path}")

    # Show statistics
    if chunks:
        avg_size = sum(len(c['text']) for c in chunks) / len(chunks)
        print(f"\nStatistics:")
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Average chunk size: {avg_size:.0f} characters")
        print(f"  First chunk preview: {chunks[0]['text'][:100]}...")

if __name__ == "__main__":
    main()
