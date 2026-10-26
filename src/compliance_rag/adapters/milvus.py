from __future__ import annotations

from typing import Any

from pymilvus import Collection, connections, utility

from compliance_rag.core.chunking import Chunk


class MilvusVectorStore:
    def __init__(self, uri: str, collection_name: str) -> None:
        self.uri = uri
        self.collection_name = collection_name

    def _connect(self) -> None:
        if not connections.has_connection("default"):
            connections.connect(alias="default", uri=self.uri)

    def ensure_collection(self) -> None:
        self._connect()
        if utility.has_collection(self.collection_name):
            return
        # Schema creation is intentionally left to the real ingestion bootstrap.
        # This keeps the adapter importable before infrastructure is available.
        raise RuntimeError(f"collection does not exist: {self.collection_name}")

    def upsert_chunk_embeddings(
        self,
        chunks: list[Chunk],
        dense_vectors: list[list[float]],
        sparse_vectors: list[dict[int, float]],
    ) -> None:
        self._connect()
        collection = Collection(self.collection_name)
        rows = [
            {
                "chunk_id": chunk.chunk_id,
                "parent_id": chunk.parent_id,
                "text": chunk.text,
                "dense_vector": dense_vectors[index],
                "sparse_vector": sparse_vectors[index],
            }
            for index, chunk in enumerate(chunks)
        ]
        collection.insert(rows)
        collection.flush()

    def search(self, query_vector: list[float], top_k: int) -> list[dict[str, Any]]:
        self._connect()
        collection = Collection(self.collection_name)
        results = collection.search(
            data=[query_vector],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
        )
        return [hit.to_dict() for hit in results[0]]
