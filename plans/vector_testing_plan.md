# SingleStore Vector Testing Framework Plan

## Overview
Create a comprehensive testing suite to evaluate vector column configurations, index types, and search strategies in SingleStore, focusing on 1024 and 1536 dimension vectors.

## Testing Dimensions

### 1. Vector Dimensions
Test embedding models with specific dimensions:
- **1024 dims**: Medium-sized embeddings
- **1536 dims**: Larger embeddings (e.g., OpenAI text-embedding-ada-002)

### 2. Index Types
Test two primary index configurations:
- **IVF_PQFS**: Inverted File with Product Quantization and Fast Scan
  - Parameters to test:
    - `nlist`: [256, 512, 1024]
    - `m`: [16, 32]
    - `nbits`: [8]
- **HNSW_FLAT**: Hierarchical Navigable Small World with Flat storage
  - Parameters to test:
    - `M`: [16, 32, 48]
    - `ef_construction`: [200, 500]
    - `ef`: [50, 100, 200]

### 3. Data Configurations
- **Small dataset**: 1K chunks (Pride & Prejudice)
- **Medium dataset**: 10K chunks (multiple books)
- **Very large dataset**: 1M+ chunks

### 4. Query Types
- **Dot Product**: Direct similarity computation
- **Top-K Search**: K-nearest neighbors retrieval

## Implementation Phases

### Phase 1: Baseline Testing (No Indexes)
**Goal**: Establish baseline performance with small dataset and no indexes

#### Setup
- Dataset: Small (1K chunks)
- Vector dimensions: 1024 and 1536
- No indexes (brute force search)

#### Tests
1. **Dot Product Queries**
   - Single vector dot product calculation
   - Measure raw computation time
   - Test both 1024 and 1536 dimensions

2. **Top-K Queries**
   - K values: [5, 10, 20, 50]
   - Measure query latency without indexes
   - Establish accuracy baseline (100% recall)

#### Metrics to Collect
- Query latency (p50, p95, p99)
- Memory usage
- CPU utilization
- Storage size for vectors

#### Deliverables
- Baseline performance report
- Raw vector operation benchmarks
- Foundation for comparison with indexed approaches

### Phase 2: Comprehensive Testing (Indexes & Scale)
**Goal**: Test index performance and scalability with diverse query types

#### Setup
- Datasets: Small (1K), Medium (10K), Very Large (1M+)
- Vector dimensions: 1024 and 1536
- Index types: IVF_PQFS and HNSW_FLAT

#### Query Types for Phase 2

##### 2.1 Core Query Types
1. **Single Vector Search**
   - Find vectors similar to one query vector
   - Measure individual query latency
   - Test with different similarity thresholds

2. **Batch Search**
   - Execute multiple queries simultaneously
   - Batch sizes: [10, 50, 100, 500]
   - Measure throughput and latency distribution
   - Test parallel vs sequential execution

3. **Top-K Search (KNN)**
   - K values: [5, 10, 20, 50, 100]
   - Measure how latency scales with K
   - Track accuracy vs baseline at each K

4. **Range Search**
   - Find all vectors within distance threshold
   - Thresholds: [0.5, 0.7, 0.9] for normalized vectors
   - Measure result set sizes and query times
   - Test selectivity impact on performance

##### 2.2 Advanced Query Types
5. **Filtered Search**
   - Vector search with WHERE clauses
   - Filter types:
     - Metadata filters (source, date, category)
     - Range filters (chunk_id, length)
     - Text pattern filters (LIKE)
   - Measure filter selectivity impact
   - Test filter pushdown effectiveness

6. **Hybrid Search**
   - Combine vector similarity and full-text scores
   - Weight combinations: [0.0, 0.3, 0.5, 0.7, 1.0]
   - Test ranking quality
   - Measure overhead of score combination

7. **Aggregation Queries**
   - Group by with vector operations
   - COUNT, AVG similarity scores
   - Top-K per group
   - Measure aggregation overhead

#### Test Matrix

##### Index Configuration Testing
For each combination of:
- Dimension: [1024, 1536]
- Dataset size: [1K, 10K, 1M]
- Index type: [IVF_PQFS, HNSW_FLAT]
- Query type: [All 7 types listed above]

Test:
```
For IVF_PQFS:
  - nlist: [256, 512, 1024]
  - m: [16, 32]
  - Run all query types

For HNSW_FLAT:
  - M: [16, 32, 48]
  - ef_construction: [200, 500]
  - ef_search: [50, 100, 200]
  - Run all query types
```

#### Scale Testing
Progressive scaling tests:
```
1. Start with 1K chunks
2. Scale to 10K chunks
3. Scale to 100K chunks
4. Scale to 1M chunks
5. Continue until performance degrades or limits reached
```

#### Metrics to Collect

##### Performance Metrics
- **Per Query Type:**
  - Latency (p50, p95, p99, max)
  - Throughput (queries/second)
  - Result set sizes
  - Time to first result

- **System Metrics:**
  - Index build time
  - Index storage overhead
  - Memory consumption
  - CPU utilization
  - Network I/O

##### Quality Metrics
- **Accuracy Metrics:**
  - Recall@K (compared to Phase 1 baseline)
  - Precision@K
  - Mean Reciprocal Rank (MRR)
  - Normalized Discounted Cumulative Gain (NDCG)

- **Hybrid Search Metrics:**
  - Relevance improvement over pure vector/text
  - Optimal weight configurations
  - Result diversity

##### Resource Metrics
- Memory consumption (index + data)
- CPU usage patterns per query type
- Storage requirements
- Cache hit rates
- Network bandwidth usage

#### Deliverables
- Comprehensive performance comparison report
- Query type performance profiles
- Optimal index configurations per query type
- Scaling recommendations
- Best practices for each query pattern

## Test Scenarios

### Scenario 1: Small Dataset Baseline (Phase 1)
```python
# No index, brute force search
For dim in [1024, 1536]:
    - Load 1K chunks with embeddings
    - No index creation
    - Run dot product queries (100 samples)
    - Run top-k queries (k=5,10,20,50)
    - Measure baseline metrics
```

### Scenario 2: Query Type Comparison (Phase 2)
```python
For dataset_size in [1K, 10K, 1M]:
    For dim in [1024, 1536]:
        For index_type in ['IVF_PQFS', 'HNSW_FLAT']:
            For query_type in [single, batch, topk, range, filtered, hybrid, aggregation]:
                - Load data
                - Create index with optimal parameters
                - Run query type tests
                - Collect metrics
                - Compare with baseline
```

### Scenario 3: Scale Limits Testing (Phase 2)
```python
For dim in [1024, 1536]:
    For index_type in ['IVF_PQFS', 'HNSW_FLAT']:
        For query_type in [single, batch, topk]:
            chunk_count = 1000
            while performance_acceptable:
                - Load chunk_count vectors
                - Build index
                - Measure query performance
                - chunk_count *= 10
```

## Results Storage

### Results Database Schema
```sql
CREATE TABLE test_results (
    test_id VARCHAR(100),
    phase INT,
    dataset_size INT,
    vector_dims INT,
    index_type VARCHAR(50),
    index_params JSON,
    query_type VARCHAR(50),
    query_params JSON,
    k_value INT,
    batch_size INT,
    filter_type VARCHAR(50),
    hybrid_weight FLOAT,
    load_time_ms FLOAT,
    index_time_ms FLOAT,
    query_latency_p50 FLOAT,
    query_latency_p95 FLOAT,
    query_latency_p99 FLOAT,
    query_latency_max FLOAT,
    throughput_qps FLOAT,
    result_count INT,
    recall_at_k FLOAT,
    precision_at_k FLOAT,
    mrr FLOAT,
    ndcg FLOAT,
    memory_usage_mb FLOAT,
    storage_size_mb FLOAT,
    cpu_usage_percent FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE phase1_baseline (
    vector_dims INT,
    query_type VARCHAR(50),
    k_value INT,
    latency_ms FLOAT,
    memory_mb FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Success Criteria

### Phase 1 Success Criteria
- Establish clear baseline metrics for both dimensions
- Document brute force performance limits
- Identify memory and CPU requirements for no-index approach

### Phase 2 Success Criteria
- Achieve <100ms p95 latency for top-k queries on medium dataset
- Maintain >90% recall@10 vs brute force baseline
- Successfully handle 1M+ vectors with acceptable performance
- Identify optimal configurations for each query type
- Document performance characteristics for all query patterns
- Provide clear guidance on query type selection

## Timeline

### Phase 1: 1 Week
- Day 1-2: Setup and data preparation
- Day 3-4: Run baseline tests
- Day 5: Analysis and reporting

### Phase 2: 3 Weeks
- Week 1: Index configuration and query type testing on small/medium datasets
- Week 2: Scale testing to 1M+ vectors with all query types
- Week 3: Analysis, optimization, and final reporting

## Deliverables

### Phase 1 Deliverables
1. Baseline performance metrics for 1024 and 1536 dimensions
2. Brute force search characteristics documentation
3. Foundation metrics for Phase 2 comparison

### Phase 2 Deliverables
1. Comprehensive index performance comparison
2. Query type performance profiles and recommendations
3. Scaling analysis and limits documentation
4. Configuration recommendation matrix per query type
5. Best practices guide for vector search in SingleStore
6. Reusable testing framework for future evaluations

## Next Steps

1. Review and approve this plan
2. Begin Phase 1 implementation
3. Use Phase 1 results to refine Phase 2 approach
4. Execute Phase 2 with insights from baseline testing