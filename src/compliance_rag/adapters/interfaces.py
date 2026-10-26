from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from compliance_rag.core.chunking import Chunk


@dataclass
class TableResult:
    page: int
    markdown: str
    confidence: float | None = None


@dataclass
class ParsedDocument:
    source_path: Path
    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    images: list[str] = field(default_factory=list)
    tables: list[TableResult] = field(default_factory=list)


class DocumentParser(Protocol):
    def parse(self, path: Path) -> ParsedDocument:
        ...


class OCRClient(Protocol):
    def extract_tables(self, image_path: Path, page: int) -> list[TableResult]:
        ...


class DocumentStore(Protocol):
    def document_exists_by_hash(self, document_hash: str) -> bool:
        ...

    def upsert_document(self, document_id: str, document: ParsedDocument, document_hash: str) -> None:
        ...

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        ...

    def mark_document_status(self, document_id: str, status: str, error: str | None = None) -> None:
        ...


class VectorStore(Protocol):
    def upsert_chunk_embeddings(
        self,
        chunks: list[Chunk],
        dense_vectors: list[list[float]],
        sparse_vectors: list[dict[int, float]],
    ) -> None:
        ...

    def search(
        self,
        query_vector: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:
        ...


class Embedder(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


class SparseEncoder(Protocol):
    def encode(self, texts: list[str]) -> list[dict[int, float]]:
        ...


class LLMClient(Protocol):
    def clean_to_markdown(self, text: str) -> str:
        ...

    def generate(self, prompt: str) -> str:
        ...

    def summarize(self, context: str, query: str) -> str:
        ...
