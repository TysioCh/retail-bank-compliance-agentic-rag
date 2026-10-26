from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    environment: str = "development"
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_database: str = "compliance_rag"
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "policy_chunks"
    artifact_root: str = "artifacts"
    ocr_backend: str = "paddleocr"
    child_token_limit: int = 256
    child_overlap_tokens: int = 50
    parent_token_limit: int = 512
    top_k_per_route: int = 50
    rrf_k: int = 60
    rrf_top_k: int = 50
    reranker_top_k: int = 5
    reranker_score_threshold: float = 0.5
    query_rewrite_enabled: bool = True
    hyde_enabled: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            environment=os.getenv("COMPLIANCE_RAG_ENV", "development"),
            mongo_uri=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
            mongo_database=os.getenv("MONGO_DATABASE", "compliance_rag"),
            milvus_uri=os.getenv("MILVUS_URI", "http://localhost:19530"),
            milvus_collection=os.getenv("MILVUS_COLLECTION", "policy_chunks"),
            artifact_root=os.getenv("ARTIFACT_ROOT", "artifacts"),
            ocr_backend=os.getenv("OCR_BACKEND", "paddleocr"),
        )
