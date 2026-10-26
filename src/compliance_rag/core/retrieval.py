from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from compliance_rag.models import DocCandidate


def dedupe_candidates(
    candidates: Iterable[DocCandidate],
    key: str = "chunk_id",
) -> list[DocCandidate]:
    """Deduplicate candidates by a stable identifier while preserving first seen order."""
    seen: set[str] = set()
    output: list[DocCandidate] = []
    for candidate in candidates:
        value = candidate.metadata.get(key) if key in candidate.metadata else getattr(candidate, key, None)
        if value is None:
            output.append(candidate)
            continue
        if value in seen:
            continue
        seen.add(value)
        output.append(candidate)
    return output


def rrf_score(ranks: Iterable[int], k: int = 60) -> float:
    """Compute reciprocal rank fusion score for one document."""
    return sum(1.0 / (k + rank) for rank in ranks)


def fuse_rankings(
    rankings: Iterable[list[Any]],
    k: int = 60,
) -> dict[Any, float]:
    """Fuse multiple ranked result lists into one score map."""
    scores: dict[Any, list[int]] = defaultdict(list)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item].append(rank)
    return {item: rrf_score(ranks, k) for item, ranks in scores.items()}


def top_k_by_score(
    scores: dict[Any, float],
    top_k: int,
) -> list[tuple[Any, float]]:
    """Sort by descending score and return top_k items."""
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]


def threshold_filter(
    candidates: Iterable[DocCandidate],
    threshold: float,
) -> list[DocCandidate]:
    """Keep candidates whose score is at least threshold."""
    return [candidate for candidate in candidates if candidate.score >= threshold]
