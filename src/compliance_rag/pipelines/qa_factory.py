from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class QAPair:
    unique_id: str
    question: str
    answer: str
    keywords: list[str] | None = None
    is_negative: bool = False
    source_doc_id: str | None = None


def expand_queries(
    qa: QAPair,
    variants: list[str],
) -> list[QAPair]:
    """Create one QA sample for each paraphrased query."""
    return [
        QAPair(
            unique_id=f"{qa.unique_id}:v{index}",
            question=variant,
            answer=qa.answer,
            keywords=qa.keywords,
            is_negative=qa.is_negative,
            source_doc_id=qa.source_doc_id,
        )
        for index, variant in enumerate(variants)
    ]


def inject_negatives(qa_pairs: list[QAPair], negatives: list[QAPair]) -> list[QAPair]:
    return [*qa_pairs, *negatives]


def split_train_test(
    qa_pairs: list[QAPair],
    train_ratio: float = 0.9,
) -> tuple[list[QAPair], list[QAPair]]:
    split_index = int(len(qa_pairs) * train_ratio)
    return qa_pairs[:split_index], qa_pairs[split_index:]


class QADatasetFactory:
    def __init__(
        self,
        filter_fn: Callable[[QAPair], bool],
        query_expander: Callable[[QAPair], list[str]],
    ) -> None:
        self.filter_fn = filter_fn
        self.query_expander = query_expander

    def build(
        self,
        initial_qa: list[QAPair],
        negatives: list[QAPair],
        train_ratio: float = 0.9,
    ) -> dict[str, list[QAPair]]:
        filtered = [qa for qa in initial_qa if self.filter_fn(qa)]
        expanded: list[QAPair] = []
        for qa in filtered:
            variants = self.query_expander(qa)
            expanded.extend(expand_queries(qa, variants))
        with_negatives = inject_negatives(expanded, negatives)
        train, test = split_train_test(with_negatives, train_ratio)
        return {"train": train, "test": test}
