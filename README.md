# Text Chunking & Vector Search Toolkit

**Attention**: The code in this repository is intended for experimental use only and is not fully tested, documented, or supported by SingleStore. Visit the [SingleStore Forums](https://www.singlestore.com/forum) to ask questions about this repository.

A comprehensive framework for text chunking, vector embeddings, and semantic search using SingleStore. Python scripts to chunk data using multiple chunking strategies and loading data into SingleStore are provided.

## Overview

This toolkit provides:
- **Multiple chunking strategies** for breaking large documents into manageable pieces
- **Vector embedding generation** for semantic search capabilities (1024 and 1536-dimensional embeddings)
- **Database integration** with SingleStore (with vectors) and OpenSearch
- **Configuration-driven setup**: all database connections use `config.json`
- **Embedding model** is automatically selected based on configured dimensions
- **Vector indexes (IVF_PQFS)** are created automatically

## Project Structure

```
chunking/
├── chunkers/           # Text chunking implementations
│   ├── text_chunker.py      # Regex-based chunking
│   ├── nltk_chunker.py      # NLTK-powered chunking
│   ├── spacy_chunker.py     # spaCy NLP chunking
│   ├── langchain_chunker.py # LangChain splitters
│   └── evaluate_chunks.py   # Quality evaluation
├── singlestore/        # SingleStore integration
│   ├── create_embeddings.py # Generate vector embeddings
│   ├── vector_search.py     # Vector similarity search
│   ├── load_chunks_s2.py    # Load data into SingleStore
│   └── ...
├── opensearch/         # OpenSearch integration
│   ├── index_chunks.py      # Index documents
│   ├── search_chunks.py     # Search interface
│   └── docker-compose.yml   # Docker setup
└── data/               # Sample data
    ├── chunks.json          # Generated chunks
    └── pride_and_prejudice.txt
```

## Installation

### 1. Clone and Setup Environment

```bash
git clone https://github.com/singlestore-labs/vector-text-chunking
cd vector-text-chunking

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Core dependencies
pip install pymysql opensearch-py

# For vector embeddings
pip install sentence-transformers torch tqdm

# For different chunking strategies
pip install nltk spacy langchain langchain-community
python -m spacy download en_core_web_sm
```

**Note:** `spacy_chunker.py` requires Python ≥3.10. If your venv uses an older Python (e.g., 3.9), spaCy will fail to install. The other chunkers (`text_chunker.py`, `nltk_chunker.py`, `langchain_chunker.py`, `json_chunker.py`) work on Python 3.9+.

### 3. Configure Database Connection

```bash
# Copy example configuration file
cp config.json.example config.json

# Edit config.json and add your connection details:
# - SingleStore host, password, database
# - Embedding dimensions (1024 or 1536)
# - Chunking configuration
```

## Quick Start

### Step 1: Generate Text Chunks

```bash
# Basic semantic chunking
python chunkers/text_chunker.py data/pride_and_prejudice.txt --output data/chunks.json

# Use different chunking strategies
python chunkers/text_chunker.py data/pride_and_prejudice.txt --strategy sentences --size 10
python chunkers/nltk_chunker.py data/pride_and_prejudice.txt --strategy entities
python chunkers/spacy_chunker.py data/pride_and_prejudice.txt --strategy semantic
```

### Step 2: Load into SingleStore with Vectors

```bash
# No environment variables needed - uses config.json

# Load chunks (pass the JSON file from Step 1)
python singlestore/load_chunks_s2.py data/chunks.json

# Generate embeddings (1024 or 1536-dimensional vectors based on config)
python singlestore/create_embeddings.py

# Search with vectors
python singlestore/vector_search.py --topk "Elizabeth's feelings about Darcy"
```

### Step 3: Vector Similarity Search

```bash
# Top-k vector search
python singlestore/vector_search.py --topk "your search query"

# Find similar chunks (by chunk ID)
python singlestore/vector_search.py --similar 100

# Verify results with text output
python singlestore/vector_search.py --topk "marriage and social class" --verify
```

## Chunking Strategies

| Strategy | Implementation | Best For | Key Features |
|----------|---------------|----------|--------------|
| **Semantic** | All scripts | RAG applications | Preserves meaning & context |
| **Sentences** | NLTK, spaCy | Search indexing | Complete thoughts |
| **Entities** | NLTK, spaCy | Information extraction | Preserves named entities |
| **Recursive** | LangChain | Hierarchical docs | Smart splitting with fallbacks |
| **Token-based** | LangChain | LLM processing | Respects token limits |
| **Chapters** | text_chunker | Books/documents | Natural document divisions |


## Example RAG Pipeline with Vectors

```python
# 1. Chunk your documents
python chunkers/text_chunker.py document.txt --strategy semantic --size 1000

# 2. Generate embeddings and store
python singlestore/create_embeddings.py

# 3. Query with natural language
python singlestore/vector_search.py --topk "specific topic or question"
```

## Security

- Configuration uses `config.json` (not tracked in git)
- No credentials in code
- Example configuration provided in `config.json.example`
- SQL identifiers (database, table names) are validated to prevent SQL injection

## Advanced Features

### Custom Embeddings
Configure different embedding models in `config.json`:
```json
{
  "embeddings": {
    "dimension": 1024,  // or 1536
    "batch_size": 32
  }
}
```

Supported models:
- **1024 dimensions**: BAAI/bge-large-en-v1.5
- **1536 dimensions**: Alibaba-NLP/gte-large-en-v1.5

## Acknowledgments

- **Sample Text**: Project Gutenberg (Pride and Prejudice)
- **Libraries**: sentence-transformers, NLTK, spaCy, LangChain
- **Databases**: SingleStore, OpenSearch
- **Embedding Models**: BAAI/bge-large-en-v1.5, Alibaba-NLP/gte-large-en-v1.5
