#!/usr/bin/env python3
"""LangChain-based text chunking with multiple splitting strategies."""

import argparse
import json
from typing import List
from pathlib import Path

# LangChain imports
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    SentenceTransformersTokenTextSplitter,
    NLTKTextSplitter,
    SpacyTextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
)

class LangChainChunker:
    def __init__(self, filepath: str):
        """Initialize with text file."""
        self.filepath = Path(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            self.text = f.read()

    def chunk_character(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """Basic character-based splitting."""
        splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n",
            length_function=len,
        )
        return splitter.split_text(self.text)

    def chunk_recursive(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """Recursive splitting with multiple separators."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        return splitter.split_text(self.text)

    def chunk_token(self, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
        """Token-based splitting using tiktoken."""
        try:
            splitter = TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            return splitter.split_text(self.text)
        except ImportError:
            print("tiktoken not installed. Install with: pip install tiktoken")
            return []

    def chunk_sentence_transformers(self, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
        """Token splitting using sentence-transformers tokenizer."""
        try:
            splitter = SentenceTransformersTokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            return splitter.split_text(self.text)
        except ImportError:
            print("sentence-transformers not installed. Install with: pip install sentence-transformers")
            return []

    def chunk_nltk_sentences(self, chunk_size: int = 1000) -> List[str]:
        """NLTK-based sentence splitting."""
        try:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                print("Downloading NLTK punkt tokenizer...")
                nltk.download('punkt')

            splitter = NLTKTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=0
            )
            return splitter.split_text(self.text)
        except ImportError:
            print("NLTK not installed. Install with: pip install nltk")
            return []

    def chunk_spacy_sentences(self, chunk_size: int = 1000, pipeline: str = "en_core_web_sm") -> List[str]:
        """spaCy-based sentence splitting."""
        try:
            splitter = SpacyTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=200,
                pipeline=pipeline
            )
            return splitter.split_text(self.text)
        except ImportError:
            print(f"spaCy not installed or model {pipeline} not found")
            print("Install with: pip install spacy")
            print(f"Download model with: python -m spacy download {pipeline}")
            return []

    def chunk_semantic(self, chunk_size: int = 1000, breakpoint_threshold: str = "percentile") -> List[str]:
        """Semantic chunking based on sentence embeddings."""
        try:
            from langchain_experimental.text_splitter import SemanticChunker
            from langchain_openai import OpenAIEmbeddings
            import os

            if not os.getenv("OPENAI_API_KEY"):
                print("Semantic chunking requires OPENAI_API_KEY environment variable")
                return []

            embeddings = OpenAIEmbeddings()
            splitter = SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type=breakpoint_threshold,
            )
            return splitter.split_text(self.text)
        except ImportError as e:
            print(f"Missing dependencies for semantic chunking: {e}")
            print("Install with: pip install langchain-experimental langchain-openai")
            return []

    def get_statistics(self, chunks: List[str]) -> dict:
        """Calculate chunk statistics."""
        if not chunks:
            return {"error": "No chunks to analyze"}

        chunk_sizes = [len(chunk) for chunk in chunks]
        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(chunk_sizes) // len(chunks),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "total_chars": sum(chunk_sizes),
            "overlap_estimate": sum(chunk_sizes) - len(self.text) if sum(chunk_sizes) > len(self.text) else 0
        }

def main():
    parser = argparse.ArgumentParser(description='LangChain text chunking')
    parser.add_argument('file', help='Text file to chunk')
    parser.add_argument('--strategy',
                       choices=['character', 'recursive', 'token', 'sentence-transformers',
                               'nltk', 'spacy', 'semantic'],
                       default='recursive',
                       help='Splitting strategy')
    parser.add_argument('--size', type=int, default=1000, help='Chunk size')
    parser.add_argument('--overlap', type=int, default=200, help='Chunk overlap')
    parser.add_argument('--preview', type=int, default=0, help='Preview first N chunks')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--output', help='Save chunks to JSON')

    args = parser.parse_args()

    chunker = LangChainChunker(args.file)

    # Select strategy
    print(f"Using {args.strategy} strategy...")

    if args.strategy == 'character':
        chunks = chunker.chunk_character(args.size, args.overlap)
    elif args.strategy == 'recursive':
        chunks = chunker.chunk_recursive(args.size, args.overlap)
    elif args.strategy == 'token':
        chunks = chunker.chunk_token(args.size, args.overlap)
    elif args.strategy == 'sentence-transformers':
        chunks = chunker.chunk_sentence_transformers(args.size, args.overlap)
    elif args.strategy == 'nltk':
        chunks = chunker.chunk_nltk_sentences(args.size)
    elif args.strategy == 'spacy':
        chunks = chunker.chunk_spacy_sentences(args.size)
    elif args.strategy == 'semantic':
        chunks = chunker.chunk_semantic(args.size)

    if not chunks:
        print("No chunks created. Check dependencies.")
        return

    print(f"✅ Created {len(chunks)} chunks")

    if args.stats:
        stats = chunker.get_statistics(chunks)
        print("\n📊 Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")

    if args.preview > 0:
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
            'statistics': chunker.get_statistics(chunks),
            'chunks': chunks
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        print(f"💾 Saved to {args.output}")

if __name__ == "__main__":
    main()
