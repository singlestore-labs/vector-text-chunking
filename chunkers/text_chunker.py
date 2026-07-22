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

"""
Text Chunking Script for Large Documents
Demonstrates various chunking strategies for processing text files.
"""

import re
import argparse
from typing import List, Generator, Tuple
import json


class TextChunker:
    def __init__(self, filepath: str):
        """Initialize the chunker with a text file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.text = f.read()
        self.filepath = filepath

    def chunk_by_characters(self, chunk_size: int = 1000, overlap: int = 0) -> Generator[str, None, None]:
        """
        Chunk text by character count with optional overlap.

        Args:
            chunk_size: Number of characters per chunk
            overlap: Number of overlapping characters between chunks
        """
        start = 0
        text_len = len(self.text)

        while start < text_len:
            end = min(start + chunk_size, text_len)
            yield self.text[start:end]
            start += chunk_size - overlap

    def chunk_by_words(self, word_count: int = 200, overlap: int = 0) -> Generator[str, None, None]:
        """
        Chunk text by word count with optional overlap.

        Args:
            word_count: Number of words per chunk
            overlap: Number of overlapping words between chunks
        """
        words = self.text.split()
        start = 0

        while start < len(words):
            end = min(start + word_count, len(words))
            yield ' '.join(words[start:end])
            start += word_count - overlap

    def chunk_by_sentences(self, sentences_per_chunk: int = 5) -> Generator[str, None, None]:
        """
        Chunk text by sentences.

        Args:
            sentences_per_chunk: Number of sentences per chunk
        """
        # Simple sentence splitting (can be improved with NLTK)
        sentences = re.split(r'(?<=[.!?])\s+', self.text)

        for i in range(0, len(sentences), sentences_per_chunk):
            chunk = sentences[i:i + sentences_per_chunk]
            if chunk:
                yield ' '.join(chunk)

    def chunk_by_paragraphs(self, paragraphs_per_chunk: int = 3) -> Generator[str, None, None]:
        """
        Chunk text by paragraphs.

        Args:
            paragraphs_per_chunk: Number of paragraphs per chunk
        """
        # Split by double newlines (common paragraph separator)
        paragraphs = re.split(r'\n\s*\n', self.text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        for i in range(0, len(paragraphs), paragraphs_per_chunk):
            chunk = paragraphs[i:i + paragraphs_per_chunk]
            if chunk:
                yield '\n\n'.join(chunk)

    def chunk_by_chapters(self) -> Generator[Tuple[str, str], None, None]:
        """
        Chunk text by chapters (for books with "Chapter" markers).
        Returns tuples of (chapter_title, chapter_text)
        """
        # Pattern for finding chapters (adjust based on book format)
        chapter_pattern = r'(Chapter\s+[IVXLCDM\d]+[^\n]*)'

        # Split by chapter headings
        parts = re.split(chapter_pattern, self.text, flags=re.IGNORECASE)

        # Skip the first part if it's before the first chapter
        start_idx = 0
        if parts[0] and not re.match(chapter_pattern, parts[0], re.IGNORECASE):
            start_idx = 1

        # Pair chapter titles with their content
        for i in range(start_idx, len(parts) - 1, 2):
            chapter_title = parts[i].strip()
            chapter_text = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if chapter_text:
                yield (chapter_title, chapter_text)

    def chunk_by_semantic_blocks(self, max_chunk_size: int = 2000) -> Generator[str, None, None]:
        """
        Chunk text by semantic blocks (paragraphs) while respecting max size.
        Tries to keep paragraphs together when possible.

        Args:
            max_chunk_size: Maximum characters per chunk
        """
        paragraphs = re.split(r'\n\s*\n', self.text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        current_chunk = []
        current_size = 0

        for paragraph in paragraphs:
            para_size = len(paragraph)

            # If adding this paragraph would exceed the limit
            if current_size + para_size > max_chunk_size and current_chunk:
                yield '\n\n'.join(current_chunk)
                current_chunk = []
                current_size = 0

            # If a single paragraph is larger than max_chunk_size, split it
            if para_size > max_chunk_size:
                if current_chunk:
                    yield '\n\n'.join(current_chunk)
                    current_chunk = []
                    current_size = 0

                # Split large paragraph by sentences
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                temp_chunk = []
                temp_size = 0

                for sentence in sentences:
                    if temp_size + len(sentence) > max_chunk_size and temp_chunk:
                        yield ' '.join(temp_chunk)
                        temp_chunk = []
                        temp_size = 0
                    temp_chunk.append(sentence)
                    temp_size += len(sentence)

                if temp_chunk:
                    yield ' '.join(temp_chunk)
            else:
                current_chunk.append(paragraph)
                current_size += para_size

        # Don't forget the last chunk
        if current_chunk:
            yield '\n\n'.join(current_chunk)

    def get_statistics(self) -> dict:
        """Get basic statistics about the text."""
        return {
            'total_characters': len(self.text),
            'total_words': len(self.text.split()),
            'total_sentences': len(re.split(r'[.!?]+', self.text)),
            'total_paragraphs': len(re.split(r'\n\s*\n', self.text)),
            'total_lines': len(self.text.splitlines())
        }


def main():
    parser = argparse.ArgumentParser(description='Chunk text files using various strategies')
    parser.add_argument('file', help='Path to the text file to chunk')
    parser.add_argument('--strategy', choices=['chars', 'words', 'sentences', 'paragraphs', 'chapters', 'semantic'],
                       default='semantic', help='Chunking strategy to use')
    parser.add_argument('--size', type=int, default=1000, help='Chunk size (interpretation depends on strategy)')
    parser.add_argument('--overlap', type=int, default=0, help='Overlap between chunks (for chars/words strategies)')
    parser.add_argument('--output', help='Output file for chunks (JSON format)')
    parser.add_argument('--stats', action='store_true', help='Show text statistics')
    parser.add_argument('--preview', type=int, default=0, help='Preview first N chunks')

    args = parser.parse_args()

    # Initialize chunker
    chunker = TextChunker(args.file)

    # Show statistics if requested
    if args.stats:
        stats = chunker.get_statistics()
        print("\n📊 Text Statistics:")
        for key, value in stats.items():
            print(f"  {key.replace('_', ' ').title()}: {value:,}")
        print()

    # Perform chunking based on strategy
    chunks = []

    if args.strategy == 'chars':
        chunks = list(chunker.chunk_by_characters(args.size, args.overlap))
        strategy_desc = f"Character-based ({args.size} chars/chunk)"
    elif args.strategy == 'words':
        chunks = list(chunker.chunk_by_words(args.size, args.overlap))
        strategy_desc = f"Word-based ({args.size} words/chunk)"
    elif args.strategy == 'sentences':
        chunks = list(chunker.chunk_by_sentences(args.size))
        strategy_desc = f"Sentence-based ({args.size} sentences/chunk)"
    elif args.strategy == 'paragraphs':
        chunks = list(chunker.chunk_by_paragraphs(args.size))
        strategy_desc = f"Paragraph-based ({args.size} paragraphs/chunk)"
    elif args.strategy == 'chapters':
        chapter_chunks = list(chunker.chunk_by_chapters())
        chunks = [f"{title}\n\n{text}" for title, text in chapter_chunks]
        strategy_desc = "Chapter-based"
    elif args.strategy == 'semantic':
        chunks = list(chunker.chunk_by_semantic_blocks(args.size))
        strategy_desc = f"Semantic blocks (max {args.size} chars/chunk)"

    print(f"\n📄 Chunking Strategy: {strategy_desc}")
    print(f"📦 Total Chunks Created: {len(chunks)}")

    if chunks:
        avg_size = sum(len(c) for c in chunks) / len(chunks)
        print(f"📏 Average Chunk Size: {avg_size:.0f} characters")
        print(f"📏 Min Chunk Size: {min(len(c) for c in chunks)} characters")
        print(f"📏 Max Chunk Size: {max(len(c) for c in chunks)} characters")

    # Preview chunks if requested
    if args.preview > 0 and chunks:
        print(f"\n👁️  Preview of first {min(args.preview, len(chunks))} chunks:\n")
        for i, chunk in enumerate(chunks[:args.preview], 1):
            preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
            print(f"--- Chunk {i} ({len(chunk)} chars) ---")
            print(preview)
            print()

    # Save to file if requested
    if args.output:
        output_data = {
            'strategy': args.strategy,
            'parameters': {
                'size': args.size,
                'overlap': args.overlap
            },
            'statistics': {
                'total_chunks': len(chunks),
                'avg_chunk_size': sum(len(c) for c in chunks) / len(chunks) if chunks else 0
            },
            'chunks': chunks
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Chunks saved to: {args.output}")


if __name__ == "__main__":
    main()
