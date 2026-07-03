#!/usr/bin/env python3
"""spaCy-based text chunker with NLP-aware strategies."""

import spacy
import argparse
import json
from typing import List, Generator
import sys

class SpacyChunker:
    def __init__(self, filepath: str, model: str = 'en_core_web_sm'):
        try:
            self.nlp = spacy.load(model)
        except OSError:
            print(f"Model {model} not found. Install with: python -m spacy download {model}")
            sys.exit(1)

        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        # Process in chunks to handle large texts
        self.nlp.max_length = len(text) + 100
        self.doc = self.nlp(text)
        self.filepath = filepath

    def chunk_by_sentences(self, n_sentences: int = 5, overlap: int = 0) -> List[str]:
        """Chunk by sentences using spaCy's sentence detection."""
        sentences = list(self.doc.sents)
        chunks = []

        for i in range(0, len(sentences), n_sentences - overlap):
            chunk_sents = sentences[i:i + n_sentences]
            if chunk_sents:
                chunks.append(' '.join(sent.text.strip() for sent in chunk_sents))

        return chunks

    def chunk_semantic(self, max_tokens: int = 500, preserve_entities: bool = True) -> List[str]:
        """Semantic chunking that preserves entities and sentence boundaries."""
        chunks = []
        current_chunk = []
        current_size = 0

        for sent in self.doc.sents:
            sent_tokens = len(sent)

            # Check if we need to preserve entities
            if preserve_entities and current_chunk:
                # Check if sentence starts with a pronoun that might reference previous content
                if sent[0].pos_ in ('PRON', 'DET') and current_size + sent_tokens <= max_tokens * 1.2:
                    current_chunk.append(sent.text.strip())
                    current_size += sent_tokens
                    continue

            if current_size + sent_tokens > max_tokens and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0

            current_chunk.append(sent.text.strip())
            current_size += sent_tokens

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def chunk_by_tokens(self, n_tokens: int = 500, overlap: int = 0) -> List[str]:
        """Chunk by exact token count."""
        tokens = [token for token in self.doc if not token.is_space]
        chunks = []

        for i in range(0, len(tokens), n_tokens - overlap):
            chunk_tokens = tokens[i:i + n_tokens]
            if chunk_tokens:
                start_idx = chunk_tokens[0].idx
                end_idx = chunk_tokens[-1].idx + len(chunk_tokens[-1].text)
                chunks.append(self.doc.text[start_idx:end_idx].strip())

        return chunks

    def chunk_preserve_entities(self, target_size: int = 1000) -> List[str]:
        """Chunk while never splitting named entities."""
        chunks = []
        current_chunk = []
        current_size = 0

        # Group sentences with their entities
        for sent in self.doc.sents:
            sent_text = sent.text.strip()
            sent_size = len(sent_text)

            # Find entities in this sentence
            sent_ents = [ent for ent in self.doc.ents if ent.start >= sent.start and ent.end <= sent.end]

            # If adding this sentence exceeds limit and we have content
            if current_size + sent_size > target_size and current_chunk:
                # Check if any entities continue into next sentence
                continue_ents = any(ent.end > sent.end for ent in sent_ents)

                if not continue_ents:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_size = 0

            current_chunk.append(sent_text)
            current_size += sent_size

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def get_stats(self) -> dict:
        """Get document statistics."""
        return {
            'sentences': len(list(self.doc.sents)),
            'tokens': len([t for t in self.doc if not t.is_space]),
            'entities': len(self.doc.ents),
            'entity_types': list(set(ent.label_ for ent in self.doc.ents))
        }

def main():
    parser = argparse.ArgumentParser(description='spaCy-based text chunker')
    parser.add_argument('file', help='Text file to chunk')
    parser.add_argument('--model', default='en_core_web_sm', help='spaCy model to use')
    parser.add_argument('--strategy', choices=['sentences', 'semantic', 'tokens', 'entities'],
                       default='semantic', help='Chunking strategy')
    parser.add_argument('--size', type=int, default=500, help='Chunk size')
    parser.add_argument('--overlap', type=int, default=0, help='Overlap for applicable strategies')
    parser.add_argument('--preserve-entities', action='store_true', help='Never split entities')
    parser.add_argument('--stats', action='store_true', help='Show document statistics')
    parser.add_argument('--preview', type=int, default=0, help='Preview first N chunks')
    parser.add_argument('--output', help='Save chunks to JSON file')

    args = parser.parse_args()

    print(f"Loading spaCy model '{args.model}'...")
    chunker = SpacyChunker(args.file, args.model)

    if args.stats:
        stats = chunker.get_stats()
        print(f"\n📊 Document Statistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    # Perform chunking
    if args.strategy == 'sentences':
        chunks = chunker.chunk_by_sentences(args.size, args.overlap)
    elif args.strategy == 'semantic':
        chunks = chunker.chunk_semantic(args.size, args.preserve_entities)
    elif args.strategy == 'tokens':
        chunks = chunker.chunk_by_tokens(args.size, args.overlap)
    elif args.strategy == 'entities':
        chunks = chunker.chunk_preserve_entities(args.size)

    print(f"\n📦 Chunks: {len(chunks)}")
    if chunks:
        print(f"📏 Avg size: {sum(len(c) for c in chunks) // len(chunks)} chars")

    if args.preview > 0:
        print(f"\n👁️  Preview:")
        for i, chunk in enumerate(chunks[:args.preview], 1):
            print(f"\n--- Chunk {i} ---")
            print(chunk[:300] + "..." if len(chunk) > 300 else chunk)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump({'chunks': chunks, 'total': len(chunks)}, f, indent=2)
        print(f"\n✅ Saved to {args.output}")

if __name__ == "__main__":
    main()