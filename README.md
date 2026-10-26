# Retail Bank Compliance Agentic RAG

面向零售银行政策与合规问答场景的 Agentic RAG 系统。

## Overview

This project builds an end-to-end retrieval-augmented generation pipeline for regulatory policy and compliance question answering. It combines document ingestion, hybrid retrieval, ranking, agentic orchestration, model fine-tuning, quantization, and serving into a modular codebase.

The included corpus consists of public regulatory documents from official sources such as the National Financial Regulatory Administration, the People's Bank of China, and government websites.

## Key Features

- PDF parsing with PyMuPDF and table OCR
- Markdown cleaning and title/image/table association
- Parent-child chunking for retrieval and generation
- Hybrid retrieval with BM25, dense embeddings, and learned sparse embeddings
- Reciprocal rank fusion and cross-encoder reranking
- Agent workflow with guardrails, routing, recovery, summarization, web search, and rejection
- QA data generation, teacher-model distillation, pointwise and SFT dataset construction
- LoRA fine-tuning, AWQ INT4 quantization, and vLLM serving adapters
- Retrieval and generation evaluation metrics

## Architecture

```text
PDF
  -> PyMuPDF + PaddleOCR PP-StructureV3
  -> LLM Markdown cleaning
  -> MongoDB
  -> Parent/Child chunking
  -> Milvus

Query
  -> Guardrail
  -> Router
      -> RAG
      -> Summary
      -> Web Search
      -> Reject
  -> Recovery
  -> Postprocessing
  -> Answer + citations + related images
```

## Repository Layout

```text
compliance-agentic-rag/
  configs/                 # runtime configuration
  samples/                 # public sample PDFs
  scripts/                 # runnable demo scripts
  src/compliance_rag/
    adapters/              # PDF, OCR, Mongo, Milvus, in-memory adapters
    core/                  # chunking, RRF, deduplication
    pipelines/             # data, training, evaluation pipelines
    services/              # retrieval, agent, and serving services
  tests/                   # unit tests
  docs/                    # architecture and status documentation
```

## Quick Start

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python scripts/demo_ingest.py
.venv/bin/python scripts/demo_query.py
.venv/bin/python -m unittest discover -s tests -v
```

The ingestion demo parses the PDFs in `samples/` and stores the resulting chunks and vectors in local memory. The query demo runs BM25, dense, and sparse retrieval, fuses them with RRF, and applies a score-based reranking pass.

## Data and Deployment

The repository ships a small set of public sample documents for reproducible examples. It does not include private data, internal knowledge bases, trained model weights, or credentials.

Production integrations for MongoDB, Milvus, and cloud storage are provided as adapters. Model training, quantization, and vLLM serving are provided as configuration adapters and documented in the respective modules.

## Evaluation

Implemented metrics include:

- Jaccard keyword similarity
- semantic answer similarity
- `Recall@K`
- `Hit@K`
- `MRR`

The evaluation module is designed to be used with a golden QA dataset and a dataset version identifier, so every reported result can be reproduced independently.
