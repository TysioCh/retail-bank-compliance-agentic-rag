from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any

from compliance_rag.adapters.interfaces import ParsedDocument
from compliance_rag.core.chunking import Chunk


class InMemoryDocumentStore:
    def __init__(self) -> None:
        self.documents: dict[str, dict[str, Any]] = {}
        self.chunks: dict[str, dict[str, Any]] = {}
        self.hashes: set[str] = set()

    def document_exists_by_hash(self, document_hash: str) -> bool:
        return document_hash in self.hashes

    def upsert_document(self, document_id: str, document: ParsedDocument, document_hash: str) -> None:
        self.hashes.add(document_hash)
        self.documents[document_id] = {
            "document_id": document_id,
            "source_path": str(document.source_path),
            "text": document.text,
            "metadata": document.metadata,
            "document_hash": document_hash,
            "images": document.images,
            "tables": [table.__dict__ for table in document.tables],
            "status": "pending",
        }

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = {
                "chunk_id": chunk.chunk_id,
                "parent_id": chunk.parent_id,
                "text": chunk.text,
                "start": chunk.start,
                "end": chunk.end,
            }

    def mark_document_status(self, document_id: str, status: str, error: str | None = None) -> None:
        if document_id not in self.documents:
            return
        self.documents[document_id]["status"] = status
        if error:
            self.documents[document_id]["error"] = error


class InMemoryVectorStore:
    def __init__(self) -> None:
        self.dense: dict[str, list[float]] = {}
        self.sparse: dict[str, dict[int, float]] = {}

    def upsert_chunk_embeddings(
        self,
        chunks: list[Chunk],
        dense_vectors: list[list[float]],
        sparse_vectors: list[dict[int, float]],
    ) -> None:
        for index, chunk in enumerate(chunks):
            self.dense[chunk.chunk_id] = dense_vectors[index]
            self.sparse[chunk.chunk_id] = sparse_vectors[index]

    def search(self, query_vector: list[float], top_k: int) -> list[dict[str, Any]]:
        scored = [
            {
                "chunk_id": chunk_id,
                "score": self._cosine(query_vector, vector),
            }
            for chunk_id, vector in self.dense.items()
        ]
        return sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)


class HashEmbedder:
    def __init__(self, dimensions: int = 128) -> None:
        self.dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = text.split() or list(text)
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            value = int.from_bytes(digest[:8], "big")
            index = value % self.dimensions
            sign = 1.0 if (value >> 63) & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class CharSparseEncoder:
    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self._next_id = 0

    def encode(self, texts: list[str]) -> list[dict[int, float]]:
        return [self._encode(text) for text in texts]

    def _encode(self, text: str) -> dict[int, float]:
        counts: dict[int, float] = defaultdict(float)
        for char in text:
            if char.isspace():
                continue
            token_id = self.vocab.setdefault(char, self._next_id)
            if token_id == self._next_id:
                self._next_id += 1
            counts[token_id] += 1.0
        return dict(counts)


class IdentityCleaner:
    def clean_to_markdown(self, text: str) -> str:
        return text

    def generate(self, prompt: str) -> str:
        return ""

    def summarize(self, context: str, query: str) -> str:
        return ""
