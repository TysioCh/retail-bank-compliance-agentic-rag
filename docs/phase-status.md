# Phase Status

## Completed

- Phase 0: project scaffold, configuration, Docker Compose, core models, retrieval utilities
- Phase 1: ingestion pipeline orchestration, idempotent hashing, Mongo/Milvus adapters, PDF/OCR adapter skeletons, ingestion service
- Phase 2: QA dataset factory, query expansion, negative injection, train/test split
- Phase 3: hybrid retrieval service, RRF, threshold and top-k logic
- Phase 4: pure Agent orchestration, recovery order, summary/web/reject routing, postprocessor interface
- Phase 5: teacher trace and SFT/pointwise data builders, training command adapters
- Phase 7: evaluation metrics

## Pending

- Wire real LLM, embedding, sparse encoder, reranker implementations
- Start MongoDB, MinIO, etcd, Milvus and validate ingestion end-to-end
- Add LangGraph runtime execution and API integration tests
- Add real PDF fixtures and golden QA set
- Add vLLM and AWQ execution scripts
- Add load testing scripts
