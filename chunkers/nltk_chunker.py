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

"""NLTK-based text chunking with linguistic features."""

import argparse
import json
import sys
from typing import List, Tuple
import re

try:
    import nltk
except ImportError:
    print("NLTK not installed. Install with: pip install nltk")
    sys.exit(1)

class NLTKChunker:
    def __init__(self, filepath: str):
        """Initialize with text and download required NLTK data."""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.text = f.read()

        # Download required NLTK data
        self._ensure_nltk_data()

    def _ensure_nltk_data(self):
        """Download required NLTK data if not present."""
        required_data = [
            ('tokenizers/punkt', 'punkt'),
            ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
            ('chunkers/maxent_ne_chunker', 'maxent_ne_chunker'),
            ('corpora/words', 'words'),
            ('tokenizers/punkt_tab', 'punkt_tab')
        ]

        for path, name in required_data:
            try:
                nltk.data.find(path)
            except LookupError:
                print(f"Downloading NLTK {name}...")
                nltk.download(name, quiet=True)

    def chunk_by_sentences(self, n_sentences: int = 5, overlap: int = 0) -> List[str]:
        """Chunk using NLTK's Punkt sentence tokenizer."""
        sentences = nltk.sent_tokenize(self.text)
        chunks = []

        step = max(1, n_sentences - overlap)
        for i in range(0, len(sentences), step):
            chunk = sentences[i:i + n_sentences]
            if chunk:
                chunks.append(' '.join(chunk))

        return chunks

    def chunk_by_tokens(self, n_tokens: int = 100, overlap: int = 0) -> List[str]:
        """Chunk by NLTK word tokens."""
        tokens = nltk.word_tokenize(self.text)
        chunks = []

        step = max(1, n_tokens - overlap)
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i:i + n_tokens]
            if chunk_tokens:
                chunks.append(' '.join(chunk_tokens))

        return chunks

    def chunk_by_noun_phrases(self, max_size: int = 1000) -> List[str]:
        """Chunk keeping noun phrases together."""
        sentences = nltk.sent_tokenize(self.text)
        chunks = []
        current_chunk = []
        current_size = 0

        # Define grammar for noun phrase extraction
        grammar = r"""
            NP: {<DT|JJ|NN.*>+}  # Noun phrase
            PP: {<IN><NP>}       # Prepositional phrase
            VP: {<VB.*><NP|PP>*} # Verb phrase
        """
        parser = nltk.RegexpParser(grammar)

        for sent in sentences:
            tokens = nltk.word_tokenize(sent)
            pos_tags = nltk.pos_tag(tokens)
            tree = parser.parse(pos_tags)

            sent_size = len(sent)

            if current_size + sent_size > max_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0

            current_chunk.append(sent)
            current_size += sent_size

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def chunk_by_named_entities(self, max_size: int = 1000) -> List[str]:
        """Chunk preserving named entities."""
        sentences = nltk.sent_tokenize(self.text)
        chunks = []
        current_chunk = []
        current_size = 0
        current_entities = set()

        for sent in sentences:
            tokens = nltk.word_tokenize(sent)
            pos_tags = nltk.pos_tag(tokens)

            # Extract named entities
            try:
                tree = nltk.ne_chunk(pos_tags, binary=False)
                entities = []
                for subtree in tree:
                    if hasattr(subtree, 'label'):
                        entity = ' '.join(word for word, tag in subtree)
                        entities.append((entity, subtree.label()))
            except:
                entities = []

            sent_size = len(sent)

            # Check if entities continue from previous chunk
            entity_names = {e[0] for e in entities}
            continues_entity = bool(current_entities & entity_names)

            if current_size + sent_size > max_size and current_chunk and not continues_entity:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
                current_entities = set()

            current_chunk.append(sent)
            current_size += sent_size
            current_entities.update(entity_names)

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def chunk_by_paragraphs(self, n_paragraphs: int = 3) -> List[str]:
        """Chunk by paragraphs using NLTK's blank line tokenizer."""
        # Use NLTK's BlanklineTokenizer
        tokenizer = nltk.tokenize.BlanklineTokenizer()
        paragraphs = tokenizer.tokenize(self.text)
        chunks = []

        for i in range(0, len(paragraphs), n_paragraphs):
            chunk = paragraphs[i:i + n_paragraphs]
            if chunk:
                chunks.append('\n\n'.join(chunk))

        return chunks

    def chunk_by_pos_patterns(self, pattern: str = None, max_size: int = 1000) -> List[str]:
        """Chunk based on POS tag patterns."""
        if pattern is None:
            # Default pattern: complete sentences/clauses
            pattern = r"""
                S: {<NP><VP>}     # Simple sentence
                NP: {<DT>?<JJ>*<NN.*>+}  # Noun phrase
                VP: {<VB.*><NP|PP>*}     # Verb phrase
                PP: {<IN><NP>}    # Prepositional phrase
            """

        parser = nltk.RegexpParser(pattern)
        sentences = nltk.sent_tokenize(self.text)
        chunks = []
        current_chunk = []
        current_size = 0

        for sent in sentences:
            tokens = nltk.word_tokenize(sent)
            pos_tags = nltk.pos_tag(tokens)
            tree = parser.parse(pos_tags)

            sent_size = len(sent)

            if current_size + sent_size > max_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0

            current_chunk.append(sent)
            current_size += sent_size

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def get_statistics(self) -> dict:
        """Get text statistics using NLTK."""
        tokens = nltk.word_tokenize(self.text)
        sentences = nltk.sent_tokenize(self.text)

        # POS tagging for first 1000 tokens (for speed)
        sample_tokens = tokens[:1000]
        pos_tags = nltk.pos_tag(sample_tokens)
        pos_counts = {}
        for word, tag in pos_tags:
            pos_counts[tag] = pos_counts.get(tag, 0) + 1

        # Named entities in first 20 sentences
        entity_types = set()
        for sent in sentences[:20]:
            sent_tokens = nltk.word_tokenize(sent)
            sent_pos = nltk.pos_tag(sent_tokens)
            try:
                tree = nltk.ne_chunk(sent_pos)
                for subtree in tree:
                    if hasattr(subtree, 'label'):
                        entity_types.add(subtree.label())
            except:
                pass

        return {
            'total_tokens': len(tokens),
            'unique_tokens': len(set(tokens)),
            'total_sentences': len(sentences),
            'avg_sentence_length': len(tokens) / len(sentences) if sentences else 0,
            'top_pos_tags': sorted(pos_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            'entity_types_found': list(entity_types)
        }

def main():
    parser = argparse.ArgumentParser(description='NLTK-based text chunking')
    parser.add_argument('file', help='Text file to chunk')
    parser.add_argument('--strategy',
                       choices=['sentences', 'tokens', 'noun_phrases', 'entities', 'paragraphs', 'pos_patterns'],
                       default='sentences',
                       help='Chunking strategy')
    parser.add_argument('--size', type=int, default=5,
                       help='Chunk size (units depend on strategy)')
    parser.add_argument('--overlap', type=int, default=0,
                       help='Overlap for sentence/token strategies')
    parser.add_argument('--preview', type=int, default=0,
                       help='Preview first N chunks')
    parser.add_argument('--stats', action='store_true',
                       help='Show NLTK-based statistics')
    parser.add_argument('--output', help='Save chunks to JSON')

    args = parser.parse_args()

    print("Initializing NLTK chunker...")
    chunker = NLTKChunker(args.file)

    if args.stats:
        stats = chunker.get_statistics()
        print("\n📊 NLTK Text Analysis:")
        for key, value in stats.items():
            if key == 'top_pos_tags':
                print(f"  Top POS tags: {', '.join(f'{tag}({count})' for tag, count in value)}")
            elif isinstance(value, float):
                print(f"  {key.replace('_', ' ').title()}: {value:.1f}")
            elif isinstance(value, list):
                print(f"  {key.replace('_', ' ').title()}: {', '.join(map(str, value))}")
            else:
                print(f"  {key.replace('_', ' ').title()}: {value}")
        print()

    # Perform chunking
    if args.strategy == 'sentences':
        chunks = chunker.chunk_by_sentences(args.size, args.overlap)
        desc = f"NLTK sentence-based ({args.size} sentences/chunk)"
    elif args.strategy == 'tokens':
        chunks = chunker.chunk_by_tokens(args.size, args.overlap)
        desc = f"NLTK token-based ({args.size} tokens/chunk)"
    elif args.strategy == 'noun_phrases':
        chunks = chunker.chunk_by_noun_phrases(args.size)
        desc = f"NLTK noun phrase preserving (max {args.size} chars)"
    elif args.strategy == 'entities':
        chunks = chunker.chunk_by_named_entities(args.size)
        desc = f"NLTK entity preserving (max {args.size} chars)"
    elif args.strategy == 'paragraphs':
        chunks = chunker.chunk_by_paragraphs(args.size)
        desc = f"NLTK paragraph-based ({args.size} paragraphs/chunk)"
    elif args.strategy == 'pos_patterns':
        chunks = chunker.chunk_by_pos_patterns(max_size=args.size)
        desc = f"NLTK POS pattern-based (max {args.size} chars)"

    print(f"📄 Strategy: {desc}")
    print(f"📦 Chunks created: {len(chunks)}")

    if chunks:
        avg_size = sum(len(c) for c in chunks) / len(chunks)
        print(f"📏 Average chunk size: {avg_size:.0f} characters")

    if args.preview > 0 and chunks:
        print(f"\n👁️  Preview of first {min(args.preview, len(chunks))} chunks:\n")
        for i, chunk in enumerate(chunks[:args.preview], 1):
            preview = chunk[:300] + "..." if len(chunk) > 300 else chunk
            print(f"--- Chunk {i} ({len(chunk)} chars) ---")
            print(preview)
            print()

    if args.output:
        output_data = {
            'strategy': args.strategy,
            'parameters': {'size': args.size, 'overlap': args.overlap},
            'total_chunks': len(chunks),
            'chunks': chunks
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"✅ Saved to {args.output}")

if __name__ == "__main__":
    main()
