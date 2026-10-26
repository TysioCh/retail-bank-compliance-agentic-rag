from __future__ import annotations

from pathlib import Path
from typing import Any

from pymongo import MongoClient

from compliance_rag.adapters.interfaces import ParsedDocument
from compliance_rag.core.chunking import Chunk


class MongoDocumentStore:
    def __init__(self, uri: str, database: str) -> None:
        self.client = MongoClient(uri)
        self.database = self.client[database]

    def document_exists_by_hash(self, document_hash: str) -> bool:
        return self.database.documents.count_documents({"document_hash": document_hash}) > 0

    def upsert_document(self, document_id: str, document: ParsedDocument, document_hash: str) -> None:
        self.database.documents.update_one(
            {"document_id": document_id},
            {
                "$set": {
                    "document_id": document_id,
                    "source_path": str(document.source_path),
                    "text": document.text,
                    "metadata": document.metadata,
                    "document_hash": document_hash,
                    "images": document.images,
                    "tables": [table.__dict__ for table in document.tables],
                    "status": "pending",
                }
            },
            upsert=True,
        )

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self.database.chunks.update_one(
                {"chunk_id": chunk.chunk_id},
                {
                    "$set": {
                        "chunk_id": chunk.chunk_id,
                        "parent_id": chunk.parent_id,
                        "text": chunk.text,
                        "start": chunk.start,
                        "end": chunk.end,
                    }
                },
                upsert=True,
            )

    def mark_document_status(self, document_id: str, status: str, error: str | None = None) -> None:
        update: dict[str, Any] = {"status": status}
        if error:
            update["error"] = error
        self.database.documents.update_one({"document_id": document_id}, {"$set": update})
