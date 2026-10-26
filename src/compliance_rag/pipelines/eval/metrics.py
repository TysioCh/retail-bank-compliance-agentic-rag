from __future__ import annotations

from collections.abc import Sequence


def jaccard_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def combined_answer_score(
    keyword_score: float,
    semantic_score: float,
    keyword_weight: float = 0.5,
) -> float:
    return keyword_weight * keyword_score + (1 - keyword_weight) * semantic_score


def recall_at_k(relevant: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(relevant) & set(retrieved[:k])) / len(set(relevant))


def hit_at_k(relevant: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    if not relevant:
        return 0.0
    return 1.0 if set(relevant) & set(retrieved[:k]) else 0.0


def mrr(relevant_sets: Sequence[Sequence[str]], retrieved_lists: Sequence[Sequence[str]]) -> float:
    if not relevant_sets:
        return 0.0
    total = 0.0
    for relevant, retrieved in zip(relevant_sets, retrieved_lists):
        for rank, item in enumerate(retrieved, start=1):
            if item in set(relevant):
                total += 1.0 / rank
                break
    return total / len(relevant_sets)
