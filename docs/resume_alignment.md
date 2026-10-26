# Resume Alignment

This document maps resume claims to repository modules.

| Resume item | Module | Status |
| --- | --- | --- |
| 零售银行政策与合规场景 | `README.md`, `samples/` | Runnable with public samples |
| 基于 QA 对构建 Pointwise 数据 | `pipelines/qa_factory.py`, `pipelines/teacher_trace.py` | Implemented core logic |
| Dense + Sparse + BM25 召回 | `services/retrieval/service.py` | Runnable demo |
| RRF 与 Reranker 精排 | `core/retrieval.py`, `services/retrieval/service.py` | Implemented and tested |
| Agentic 路由与渐进式自救 | `services/agent/workflow.py` | Implemented and tested |
| AWQ 量化与 vLLM 部署 | `pipelines/training/quantize.py`, `docs/architecture.md` | Command adapters |
| 评测指标 | `pipelines/eval/metrics.py` | Implemented and tested |

Private data, original model weights, and non-reproducible offline metrics are not included.
