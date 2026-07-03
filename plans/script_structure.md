# Vector Testing Script Structure

## Directory Layout
```
vector_testing/
├── config/
│   ├── config.yaml           # Main configuration
│   ├── models.yaml           # Embedding model configs
│   └── queries.yaml          # Query templates
├── phase1/
│   ├── baseline_tester.py    # No-index baseline tests
│   ├── metrics_collector.py  # Phase 1 metrics
│   └── report_generator.py   # Baseline report
├── phase2/
│   ├── index_tester.py       # Index configuration tests
│   ├── scale_tester.py       # Scalability tests
│   ├── query_tester.py       # All query types
│   ├── optimizer.py          # Find optimal configs
│   └── report_generator.py   # Comprehensive report
├── shared/
│   ├── base_tester.py        # Abstract base class
│   ├── data_loader.py        # Dataset management
│   ├── embedding_generator.py # Generate embeddings
│   ├── metrics.py            # Metrics collection
│   ├── db_connector.py       # SingleStore connection
│   └── utils.py              # Utility functions
├── results/
│   ├── phase1/               # Phase 1 results
│   └── phase2/               # Phase 2 results
└── main.py                   # Main orchestrator
```

## Configuration Files

### config/config.yaml
```yaml
database:
  host: "svc-xxx.singlestore.com"
  port: 3306
  user: "admin"
  password: "${DB_PASSWORD}"  # Environment variable
  database: "vector_tests"

testing:
  dimensions: [1024, 1536]
  dataset_sizes:
    small: 1000
    medium: 10000
    large: 1000000

phase1:
  dataset: "small"
  query_samples: 100
  k_values: [5, 10, 20, 50]

phase2:
  datasets: ["small", "medium", "large"]
  index_types:
    IVF_PQFS:
      nlist: [256, 512, 1024]
      m: [16, 32]
      nbits: 8
    HNSW_FLAT:
      M: [16, 32, 48]
      ef_construction: [200, 500]
      ef: [50, 100, 200]
  query_types:
    - single
    - batch
    - topk
    - range
    - filtered
    - hybrid
    - aggregation
```

### config/queries.yaml
```yaml
single_vector:
  description: "Single vector similarity search"
  sql_template: |
    SELECT chunk_id, DOT_PRODUCT(embedding, :query_vec) as score
    FROM {table_name}
    ORDER BY score DESC
    LIMIT {limit}

topk:
  description: "Top-K nearest neighbors"
  sql_template: |
    SELECT chunk_id, DOT_PRODUCT(embedding, :query_vec) as score
    FROM {table_name}
    ORDER BY score DESC
    LIMIT {k}

range:
  description: "Range similarity search"
  sql_template: |
    SELECT chunk_id, DOT_PRODUCT(embedding, :query_vec) as score
    FROM {table_name}
    WHERE DOT_PRODUCT(embedding, :query_vec) > {threshold}

filtered:
  description: "Vector search with filters"
  sql_template: |
    SELECT chunk_id, DOT_PRODUCT(embedding, :query_vec) as score
    FROM {table_name}
    WHERE {filter_condition}
    ORDER BY score DESC
    LIMIT {limit}
```

## Core Components

### shared/base_tester.py
```python
from abc import ABC, abstractmethod
import time
import psutil
import pymysql
from typing import Dict, List, Any

class BaseTester(ABC):
    def __init__(self, config: Dict):
        self.config = config
        self.connection = None
        self.metrics = {}

    def connect(self):
        """Establish database connection"""

    @abstractmethod
    def setup_table(self, dimension: int):
        """Create table with vector column"""

    @abstractmethod
    def load_data(self, dataset_size: str):
        """Load test data"""

    @abstractmethod
    def run_test(self):
        """Execute test scenario"""

    def collect_metrics(self) -> Dict:
        """Collect performance metrics"""

    def cleanup(self):
        """Clean up resources"""
```

### phase1/baseline_tester.py
```python
from shared.base_tester import BaseTester

class BaselineTester(BaseTester):
    def setup_table(self, dimension: int):
        """Create table without indexes"""

    def run_dot_product_test(self, query_vectors: List):
        """Test dot product queries"""

    def run_topk_test(self, query_vectors: List, k_values: List[int]):
        """Test top-k queries without index"""

    def establish_baseline(self):
        """Create baseline metrics for comparison"""
```

### phase2/index_tester.py
```python
class IndexTester(BaseTester):
    def create_index(self, index_type: str, params: Dict):
        """Create vector index with specified parameters"""

    def test_index_configuration(self, index_type: str, params: Dict):
        """Test specific index configuration"""

    def compare_with_baseline(self, baseline_metrics: Dict):
        """Compare indexed performance with baseline"""
```

### phase2/query_tester.py
```python
class QueryTester:
    def test_single_vector(self, query_vec):
        """Test single vector search"""

    def test_batch(self, query_vectors, batch_size):
        """Test batch queries"""

    def test_topk(self, query_vec, k_values):
        """Test k-NN search"""

    def test_range(self, query_vec, threshold):
        """Test range search"""

    def test_filtered(self, query_vec, filter_conditions):
        """Test filtered vector search"""

    def test_hybrid(self, query_vec, text_query, weights):
        """Test hybrid vector + text search"""

    def test_aggregation(self, group_by_field):
        """Test aggregation with vector operations"""
```

### shared/metrics.py
```python
import numpy as np
from typing import List, Dict

class MetricsCollector:
    def calculate_recall_at_k(self, results: List, ground_truth: List, k: int):
        """Calculate recall@k"""

    def calculate_precision_at_k(self, results: List, ground_truth: List, k: int):
        """Calculate precision@k"""

    def calculate_mrr(self, results: List, ground_truth: List):
        """Calculate Mean Reciprocal Rank"""

    def calculate_ndcg(self, results: List, ground_truth: List):
        """Calculate NDCG"""

    def calculate_latency_percentiles(self, latencies: List[float]):
        """Calculate p50, p95, p99"""

    def measure_resource_usage(self):
        """Measure CPU, memory, I/O"""
```

## Main Orchestrator

### main.py
```python
import argparse
from phase1.baseline_tester import BaselineTester
from phase2.index_tester import IndexTester
from phase2.query_tester import QueryTester

def run_phase1(config):
    """Execute Phase 1 baseline tests"""
    tester = BaselineTester(config)
    for dim in config['testing']['dimensions']:
        tester.setup_table(dim)
        tester.load_data('small')
        tester.run_test()
        tester.save_baseline()

def run_phase2(config, baseline):
    """Execute Phase 2 comprehensive tests"""
    for dataset_size in config['phase2']['datasets']:
        for dim in config['testing']['dimensions']:
            for index_type, params in config['phase2']['index_types'].items():
                tester = IndexTester(config)
                tester.setup_table(dim)
                tester.load_data(dataset_size)
                tester.create_index(index_type, params)

                query_tester = QueryTester(tester.connection)
                for query_type in config['phase2']['query_types']:
                    query_tester.run_query_type_test(query_type)

                tester.compare_with_baseline(baseline)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['1', '2', 'all'], default='all')
    parser.add_argument('--config', default='config/config.yaml')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    if args.phase in ['1', 'all']:
        run_phase1(config)

    if args.phase in ['2', 'all']:
        baseline = load_baseline_results()
        run_phase2(config, baseline)
```

## Usage Examples

```bash
# Run Phase 1 only
python main.py --phase 1

# Run Phase 2 with custom config
python main.py --phase 2 --config custom_config.yaml

# Run specific test
python phase1/baseline_tester.py --dimension 1024 --queries 100

# Run index comparison
python phase2/index_tester.py --type IVF_PQFS --dataset medium

# Generate reports
python phase1/report_generator.py --output results/phase1_report.html
python phase2/report_generator.py --output results/phase2_report.html
```