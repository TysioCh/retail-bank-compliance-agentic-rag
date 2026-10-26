from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

from compliance_rag.adapters.interfaces import (
    DocumentParser,
    DocumentStore,
    Embedder,
    LLMClient,
    OCRClient,
    ParsedDocument,
    SparseEncoder,
    VectorStore,
)
from compliance_rag.core.chunking import split_parent_child
from compliance_rag.errors import IngestionError
from compliance_rag.models import PipelineStatus


class Clock(Protocol):
    def utcnow(self) -> str:
        ...


class SystemClock:
    def utcnow(self) -> str:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat()


def compute_document_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class IngestionPipeline:
    def __init__(
        self,
        parser: DocumentParser,
        ocr: OCRClient,
        store: DocumentStore,
        vector_store: VectorStore,
        embedder: Embedder,
        sparse_encoder: SparseEncoder,
        cleaner: LLMClient,
        parent_token_limit: int = 512,
        child_token_limit: int = 256,
        child_overlap_tokens: int = 50,
        clock: Clock | None = None,
    ) -> None:
        self.parser = parser
        self.ocr = ocr
        self.store = store
        self.vector_store = vector_store
        self.embedder = embedder
        self.sparse_encoder = sparse_encoder
        self.cleaner = cleaner
        self.parent_token_limit = parent_token_limit
        self.child_token_limit = child_token_limit
        self.child_overlap_tokens = child_overlap_tokens
        self.clock = clock or SystemClock()

    def ingest_file(self, path: Path) -> dict[str, object]:
        if not path.exists():
            raise IngestionError(f"source file does not exist: {path}")

        raw_bytes = path.read_bytes()
        document_hash = compute_document_hash(raw_bytes)

        if self.store.document_exists_by_hash(document_hash):
            return {
                "status": PipelineStatus.SUCCEEDED.value,
                "document_id": None,
                "document_hash": document_hash,
                "deduplicated": True,
            }

        document = self.parser.parse(path)
        cleaned_text = self.cleaner.clean_to_markdown(document.text)
        document.text = cleaned_text
        document.metadata["document_hash"] = document_hash
        document.metadata["ingested_at"] = self.clock.utcnow()

        self.store.upsert_document(document.document_id, document, document_hash)

        chunks = split_parent_child(
            cleaned_text,
            parent_id=document.document_id,
            parent_token_limit=self.parent_token_limit,
            child_token_limit=self.child_token_limit,
            child_overlap_tokens=self.child_overlap_tokens,
        )
        self.store.upsert_chunks(chunks)

        dense_vectors = self.embedder.embed_texts([chunk.text for chunk in chunks])
        sparse_vectors = self.sparse_encoder.encode([chunk.text for chunk in chunks])
        self.vector_store.upsert_chunk_embeddings(chunks, dense_vectors, sparse_vectors)

        self.store.mark_document_status(document.document_id, PipelineStatus.SUCCEEDED.value)
        return {
            "status": PipelineStatus.SUCCEEDED.value,
            "document_id": document.document_id,
            "document_hash": document_hash,
            "chunk_count": len(chunks),
            "deduplicated": False,
        }


def parse_document_with_ocr(
    parser: DocumentParser,
    ocr: OCRClient,
    path: Path,
) -> ParsedDocument:
    document = parser.parse(path)
    # In the real pipeline, the parser will emit image paths and the OCR
    # client will replace complex table regions. This helper keeps the
    # orchestration explicit and testable.
    for image in document.images:
        tables = ocr.extract_tables(Path(image), page=0)
        document.tables.extend(tables)
    return document
