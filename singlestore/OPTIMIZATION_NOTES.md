# Embedding Generation Optimizations

## Current Performance (Baseline)
- **Script**: `create_embeddings.py`
- **Speed**: ~16-17 chunks/second (CPU)
- **10k chunks**: ~10 minutes
- **1M chunks estimate**: ~16-17 hours

## Optimized Version Features
- **Script**: `create_embeddings_optimized.py`

### Key Optimizations

1. **Larger Batch Sizes**
   - Baseline: 32 chunks per batch
   - Optimized: 128 chunks (CPU), 256 chunks (GPU)
   - Expected improvement: 2-3x faster processing

2. **Bulk Database Operations**
   - Baseline: Individual UPDATE per chunk
   - Optimized: Bulk INSERT ... ON DUPLICATE KEY UPDATE
   - Batches of 100 records per database operation
   - Expected improvement: 5-10x faster database writes

3. **GPU Support**
   - Automatically detects and uses CUDA if available
   - GPU can provide 5-10x speedup for embedding generation
   - Larger batch sizes on GPU (256 vs 128)

4. **Progress Checkpointing**
   - Saves progress every 1000 chunks
   - Can resume from interruption
   - Prevents losing hours of work

5. **Optimized Database Commits**
   - Batch commits instead of per-record
   - Reduces network overhead
   - Connection reuse

## Expected Performance Improvements

### CPU Only
- **Optimized speed**: ~40-60 chunks/second
- **1M chunks**: ~5-7 hours (vs 16-17 hours baseline)

### With GPU (if available)
- **Optimized speed**: ~150-300 chunks/second
- **1M chunks**: ~1-2 hours

## Usage Examples

### Basic usage (resume from checkpoint if interrupted):
```bash
python create_embeddings_optimized.py
```

### Process with custom batch size:
```bash
python create_embeddings_optimized.py --batch-size 256
```

### Process limited number for testing:
```bash
python create_embeddings_optimized.py --max-chunks 1000
```

### Start fresh (ignore checkpoint):
```bash
python create_embeddings_optimized.py --no-resume
```

### Use 1536-dimensional embeddings:
```bash
python create_embeddings_optimized.py --dimension 1536
```

## Memory Considerations

- **Baseline**: ~2-3GB RAM for model + small batches
- **Optimized CPU**: ~4-5GB RAM (larger batches)
- **Optimized GPU**: ~6-8GB VRAM + 3-4GB RAM

## Monitoring Progress

The optimized script provides:
- Real-time chunks/second rate
- ETA for completion
- Progress bar with percentage
- Checkpoint saves for resumability

## For 1 Million Chunks

Recommended approach:
1. Use optimized script
2. Run overnight or in background
3. Script will checkpoint progress
4. Can resume if interrupted

Example command:
```bash
nohup python create_embeddings_optimized.py > embeddings.log 2>&1 &
```

## Database Optimization Tips

For even better performance on SingleStore:
1. Ensure sufficient compute resources on SingleStore cluster
2. Consider increasing connection pool size
3. Monitor database CPU/memory during bulk inserts
4. Could partition table for parallel processing