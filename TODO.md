# TODOs

- [ ] Make table name a param
- [ ] Think about removing the full text index … or at least comment it out for now
- [ ] What are the vector and hybrid search functions doing in create_embedding
- [ ] Fix hybrid search test (consider with removing full-text index)
- [ ] Generate 1536-dimensional embeddings as well (for comparison)?
- [ ] Start testing the vector search performance (Phase 1 of your testing plan)?
- [ ] Create vector indexes (IVF_PQFS or HNSW_FLAT)?
- [ ] load_chunks creates a f-t index which we don't want for vector testing
