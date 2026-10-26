# Evaluation

## Reproducible evaluation path

1. Place public PDFs in `samples/`.
2. Run `scripts/demo_ingest.py`.
3. Generate or load a small golden QA set.
4. Run the retrieval/answer evaluator.
5. Report metrics with dataset version and command hash.

## Metrics

- `Jaccard similarity`
- `semantic similarity`
- `Recall@K`
- `Hit@K`
- `MRR`
- optional RAGAs metrics

The original project's offline numbers are not claimed as reproducible by this repository. They are background context only.
