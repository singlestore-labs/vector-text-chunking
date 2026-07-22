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

"""Evaluate chunking quality with various metrics."""

import json
import re
import statistics
from typing import List, Dict
import argparse

class ChunkEvaluator:
    def __init__(self, chunks: List[str], original_text: str = None):
        self.chunks = chunks
        self.original_text = original_text

    def size_metrics(self) -> Dict:
        """Analyze chunk size distribution."""
        sizes = [len(chunk) for chunk in self.chunks]

        mean_val = statistics.mean(sizes) if sizes else 0
        return {
            'count': len(self.chunks),
            'mean_size': mean_val,
            'median_size': statistics.median(sizes),
            'std_dev': statistics.stdev(sizes) if len(sizes) > 1 else 0,
            'min_size': min(sizes),
            'max_size': max(sizes),
            'size_variance_coefficient': (statistics.stdev(sizes) / mean_val
                                         if mean_val > 0 and len(sizes) > 1 else 0)
        }

    def boundary_quality(self) -> Dict:
        """Check quality of chunk boundaries."""
        complete_sentences = 0
        starts_with_capital = 0
        ends_with_punctuation = 0
        broken_quotes = 0
        broken_parentheses = 0

        for chunk in self.chunks:
            chunk = chunk.strip()
            if not chunk:
                continue

            # Check sentence boundaries
            if chunk[0].isupper():
                starts_with_capital += 1
            if chunk[-1] in '.!?':
                ends_with_punctuation += 1
            if chunk[0].isupper() and chunk[-1] in '.!?':
                complete_sentences += 1

            # Check for broken quotes
            quote_count = chunk.count('"') + chunk.count("'")
            if quote_count % 2 != 0:
                broken_quotes += 1

            # Check for broken parentheses
            if chunk.count('(') != chunk.count(')'):
                broken_parentheses += 1

        total = len(self.chunks)
        return {
            'complete_sentences_ratio': complete_sentences / total if total > 0 else 0,
            'starts_with_capital_ratio': starts_with_capital / total if total > 0 else 0,
            'ends_with_punctuation_ratio': ends_with_punctuation / total if total > 0 else 0,
            'broken_quotes_ratio': broken_quotes / total if total > 0 else 0,
            'broken_parentheses_ratio': broken_parentheses / total if total > 0 else 0
        }

    def semantic_coherence(self) -> Dict:
        """Analyze semantic coherence of chunks."""
        pronoun_pattern = r'\b(he|she|it|they|this|that|these|those)\b'
        entity_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'

        chunks_with_pronouns = 0
        chunks_with_entities = 0
        avg_sentences_per_chunk = []

        for chunk in self.chunks:
            # Count pronouns
            if re.search(pronoun_pattern, chunk, re.IGNORECASE):
                chunks_with_pronouns += 1

            # Count entities
            entities = re.findall(entity_pattern, chunk)
            if entities:
                chunks_with_entities += 1

            # Count sentences
            sentences = re.split(r'[.!?]+', chunk)
            sentences = [s.strip() for s in sentences if s.strip()]
            avg_sentences_per_chunk.append(len(sentences))

        return {
            'chunks_with_pronouns_ratio': chunks_with_pronouns / len(self.chunks) if self.chunks else 0,
            'chunks_with_entities_ratio': chunks_with_entities / len(self.chunks) if self.chunks else 0,
            'avg_sentences_per_chunk': statistics.mean(avg_sentences_per_chunk) if avg_sentences_per_chunk else 0
        }

    def overlap_analysis(self) -> Dict:
        """Analyze overlap between consecutive chunks."""
        if len(self.chunks) < 2:
            return {'overlap_detected': False}

        overlaps = []
        for i in range(len(self.chunks) - 1):
            chunk1_words = set(self.chunks[i].split()[-20:])  # Last 20 words
            chunk2_words = set(self.chunks[i + 1].split()[:20])  # First 20 words

            overlap = len(chunk1_words & chunk2_words)
            overlaps.append(overlap)

        return {
            'avg_word_overlap': statistics.mean(overlaps) if overlaps else 0,
            'max_word_overlap': max(overlaps) if overlaps else 0,
            'chunks_with_overlap': sum(1 for o in overlaps if o > 0)
        }

    def information_density(self) -> Dict:
        """Measure information density in chunks."""
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were'}

        densities = []
        for chunk in self.chunks:
            words = chunk.lower().split()
            if not words:
                continue

            content_words = [w for w in words if w not in stop_words and len(w) > 2]
            density = len(content_words) / len(words) if words else 0
            densities.append(density)

        return {
            'avg_information_density': statistics.mean(densities) if densities else 0,
            'min_information_density': min(densities) if densities else 0,
            'max_information_density': max(densities) if densities else 0
        }

    def overall_score(self) -> Dict:
        """Calculate overall quality score."""
        size = self.size_metrics()
        boundary = self.boundary_quality()
        semantic = self.semantic_coherence()
        density = self.information_density()

        # Weight different aspects
        score = 0
        score += (1 - size['size_variance_coefficient']) * 20  # Consistency
        score += boundary['complete_sentences_ratio'] * 30  # Boundary quality
        score += boundary['ends_with_punctuation_ratio'] * 10
        score += (1 - boundary['broken_quotes_ratio']) * 10
        score += (1 - boundary['broken_parentheses_ratio']) * 10
        score += density['avg_information_density'] * 20

        return {
            'overall_score': min(100, max(0, score)),
            'size_consistency': (1 - size['size_variance_coefficient']) * 100,
            'boundary_quality': boundary['complete_sentences_ratio'] * 100,
            'information_density': density['avg_information_density'] * 100
        }

def main():
    parser = argparse.ArgumentParser(description='Evaluate chunking quality')
    parser.add_argument('input', help='JSON file with chunks or text file')
    parser.add_argument('--original', help='Original text file (optional)')
    parser.add_argument('--detailed', action='store_true', help='Show detailed metrics')

    args = parser.parse_args()

    # Load chunks
    if args.input.endswith('.json'):
        with open(args.input, 'r') as f:
            data = json.load(f)
            chunks = data.get('chunks', [])
    else:
        with open(args.input, 'r') as f:
            text = f.read()
            # Simple paragraph splitting for demo
            chunks = text.split('\n\n')

    # Load original if provided
    original = None
    if args.original:
        with open(args.original, 'r') as f:
            original = f.read()

    evaluator = ChunkEvaluator(chunks, original)

    print("📊 Chunk Quality Evaluation\n")

    # Size metrics
    size_metrics = evaluator.size_metrics()
    print("📏 Size Metrics:")
    print(f"  Total chunks: {size_metrics['count']}")
    print(f"  Average size: {size_metrics['mean_size']:.0f} chars")
    print(f"  Size variance coefficient: {size_metrics['size_variance_coefficient']:.2f}")

    # Boundary quality
    boundary = evaluator.boundary_quality()
    print("\n✂️  Boundary Quality:")
    print(f"  Complete sentences: {boundary['complete_sentences_ratio']:.1%}")
    print(f"  Broken quotes: {boundary['broken_quotes_ratio']:.1%}")
    print(f"  Broken parentheses: {boundary['broken_parentheses_ratio']:.1%}")

    # Semantic coherence
    semantic = evaluator.semantic_coherence()
    print("\n🧩 Semantic Coherence:")
    print(f"  Chunks with entities: {semantic['chunks_with_entities_ratio']:.1%}")
    print(f"  Avg sentences/chunk: {semantic['avg_sentences_per_chunk']:.1f}")

    # Information density
    density = evaluator.information_density()
    print("\n📈 Information Density:")
    print(f"  Average: {density['avg_information_density']:.1%}")

    # Overall score
    score = evaluator.overall_score()
    print("\n⭐ Overall Quality Score: {:.1f}/100".format(score['overall_score']))
    print(f"  Size consistency: {score['size_consistency']:.1f}/100")
    print(f"  Boundary quality: {score['boundary_quality']:.1f}/100")
    print(f"  Information density: {score['information_density']:.1f}/100")

    if args.detailed:
        print("\n📋 Detailed Metrics:")
        all_metrics = {
            'size': size_metrics,
            'boundary': boundary,
            'semantic': semantic,
            'density': density,
            'overlap': evaluator.overlap_analysis()
        }
        print(json.dumps(all_metrics, indent=2))

if __name__ == "__main__":
    main()
