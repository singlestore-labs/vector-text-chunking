# SingleStore Setup for Chunk Storage

## Installation

### 1. Install Python dependencies
```bash
source chunking-env/bin/activate
pip install pymysql
```

### 2. Connect to your SingleStore instance

#### Option A: Local SingleStore (Docker)
```bash
# Run SingleStore locally
docker run -d --name singlestore \
  -e ROOT_PASSWORD=$SINGLESTORE_PASSWORD \
  -p 3306:3306 -p 8080:8080 \
  singlestore/cluster-in-a-box
```

#### Option B: SingleStore Cloud
Use your cloud connection details (host, port, user, password)

## Usage

### 1. Insert chunks into SingleStore
```bash
# Set environment variable first
export SINGLESTORE_PASSWORD='your_actual_password'

# Basic insert (local SingleStore)
python singlestore_chunks.py insert chunks.json --password $SINGLESTORE_PASSWORD

# With cloud connection
python singlestore_chunks.py insert chunks.json \
  --host your-host.singlestore.com \
  --port 3306 \
  --user admin \
  --password $SINGLESTORE_PASSWORD \
  --database chunks_db
```

### 2. Search chunks
```bash
# Full-text search
python singlestore_chunks.py search "Elizabeth Bennet" --type fulltext

# LIKE search (slower but more flexible)
python singlestore_chunks.py search "Elizabeth" --type like --limit 10
```

### 3. View statistics
```bash
python singlestore_chunks.py stats
```

## Features

- **Full-text search**: Fast search using SingleStore's FULLTEXT index
- **Relevance scoring**: Results ranked by relevance
- **Multiple strategies**: Store chunks from different chunking strategies
- **Statistics**: Track chunk counts, sizes, and distributions

## Example Workflow

```bash
# 1. Create chunks with different strategies
python text_chunker.py pride_and_prejudice.txt --strategy sentences --size 5 --output sent_chunks.json
python text_chunker.py pride_and_prejudice.txt --strategy semantic --size 1000 --output sem_chunks.json

# 2. Insert both into SingleStore
python singlestore_chunks.py insert sent_chunks.json --password $SINGLESTORE_PASSWORD
python singlestore_chunks.py insert sem_chunks.json --password $SINGLESTORE_PASSWORD

# 3. Search across all chunks
python singlestore_chunks.py search "Darcy proposes" --type fulltext

# 4. Check statistics
python singlestore_chunks.py stats
```

## Advanced Features

### Custom SQL queries
Connect directly to query chunks:
```python
import pymysql
import os

conn = pymysql.connect(
    host='localhost',
    user='root',
    password=os.environ.get('SINGLESTORE_PASSWORD'),
    database='chunks_db'
)

cursor = conn.cursor()
cursor.execute("""
    SELECT text
    FROM chunks
    WHERE strategy = 'semantic'
    AND length > 500
    LIMIT 5
""")
results = cursor.fetchall()
```

### Vector embeddings (future enhancement)
SingleStore supports vector operations for semantic search using embeddings.