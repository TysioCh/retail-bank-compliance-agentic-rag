# Architecture

This document describes the first-pass system architecture.

## Modules

1. Ingestion
2. Data Factory
3. Retrieval
4. Agent Orchestration
5. Training and Serving
6. Evaluation

## Data lineage

```text
source_pdf
  -> parsed_document
  -> cleaned_markdown
  -> chunks
  -> embeddings
  -> qa_pairs
  -> teacher_trace
  -> pointwise_samples
  -> sft_samples
  -> eval_cases
```
