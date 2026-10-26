from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QueryRoute(str, Enum):
    RAG = "rag"
    SUMMARY = "summary"
    WEB_SEARCH = "web_search"
    REJECT = "reject"


class PipelineStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class DocCandidate:
    chunk_id: str
    doc_id: str
    parent_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedContext:
    query: str
    candidates: list[DocCandidate]
    route: QueryRoute


@dataclass
class AgentOutput:
    answer: str
    citations: list[str] = field(default_factory=list)
    related_images: list[str] = field(default_factory=list)
    route: QueryRoute | None = None
    trace: dict[str, Any] = field(default_factory=dict)
