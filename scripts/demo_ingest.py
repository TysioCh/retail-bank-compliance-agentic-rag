from __future__ import annotations

from pathlib import Path

from compliance_rag.adapters.in_memory import (
    CharSparseEncoder,
    HashEmbedder,
    IdentityCleaner,
    InMemoryDocumentStore,
    InMemoryVectorStore,
)
from compliance_rag.adapters.pdf_parser import PyMuPdfParser
from compliance_rag.pipelines.ingestion import IngestionPipeline


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    samples = root / "samples"
    pipeline = IngestionPipeline(
        parser=PyMuPdfParser(image_dir=root / "artifacts" / "images"),
        ocr=None,  # type: ignore[arg-type]
        store=InMemoryDocumentStore(),
        vector_store=InMemoryVectorStore(),
        embedder=HashEmbedder(),
        sparse_encoder=CharSparseEncoder(),
        cleaner=IdentityCleaner(),
    )

    for pdf in sorted(samples.glob("*.pdf")):
        result = pipeline.ingest_file(pdf)
        print(result)


if __name__ == "__main__":
    main()
