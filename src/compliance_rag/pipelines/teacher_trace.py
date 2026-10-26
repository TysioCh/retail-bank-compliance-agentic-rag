from __future__ import annotations

from dataclasses import dataclass

from compliance_rag.models import DocCandidate


@dataclass
class TeacherTrace:
    query: str
    merged_docs: list[DocCandidate]
    context: list[DocCandidate]
    response: str
    citations: list[str]


@dataclass
class PointwiseSample:
    query: str
    doc_id: str
    doc_text: str
    label: int
    difficulty: str


@dataclass
class SftSample:
    instruction: str
    input: str
    output: str


def build_pointwise_samples(trace: TeacherTrace) -> list[PointwiseSample]:
    """Build simple positive/negative samples from a teacher ranking.

    The real pipeline should also sample decision-boundary cases with an LLM
    judge. This function implements the stable head/tail baseline.
    """
    samples: list[PointwiseSample] = []
    if not trace.merged_docs:
        return samples

    samples.append(
        PointwiseSample(
            query=trace.query,
            doc_id=trace.merged_docs[0].chunk_id,
            doc_text=trace.merged_docs[0].text,
            label=1,
            difficulty="easy_positive",
        )
    )
    samples.append(
        PointwiseSample(
            query=trace.query,
            doc_id=trace.merged_docs[-1].chunk_id,
            doc_text=trace.merged_docs[-1].text,
            label=0,
            difficulty="easy_negative",
        )
    )
    return samples


def build_sft_sample(
    trace: TeacherTrace,
    system_prompt: str,
) -> SftSample:
    context = "\n\n".join(
        f"[{index + 1}] {candidate.text}" for index, candidate in enumerate(trace.context)
    )
    instruction = f"{system_prompt}\n\nQuery: {trace.query}\n\nContext:\n{context}"
    output = trace.response
    if trace.citations:
        citation_text = "".join(f"【引用{cite}】" for cite in trace.citations)
        output = f"{output}\n{citation_text}"
    return SftSample(instruction=instruction, input="", output=output)
