from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compliance_rag.adapters.in_memory import (
    CharSparseEncoder,
    HashEmbedder,
    IdentityCleaner,
    InMemoryDocumentStore,
    InMemoryVectorStore,
)
from compliance_rag.adapters.pdf_parser import PyMuPdfParser
from compliance_rag.models import AgentOutput, DocCandidate, QueryRoute
from compliance_rag.pipelines.ingestion import IngestionPipeline
from compliance_rag.services.retrieval.service import HybridRetrievalService
from scripts.demo_query import (
    BM25Retriever,
    DenseRetriever,
    ScoreReranker,
    SparseRetriever,
)


class ExtractiveAnswerGenerator:
    """Build a citation-grounded answer from retrieved passages.

    This is a deterministic placeholder used by the local demo. The production
    path should replace it with Qwen3-8B via the agent service.
    """

    def generate(self, query: str, candidates: list[DocCandidate]) -> AgentOutput:
        if not candidates:
            return AgentOutput(answer="无答案", route=QueryRoute.RAG)

        selected = candidates[:3]
        passages = "\n".join(
            f"- {candidate.text.strip()[:180]}" for candidate in selected
        )
        answer = f"根据检索结果，与问题相关的条款包括：\n{passages}"
        citations = [candidate.chunk_id for candidate in selected]
        return AgentOutput(
            answer=answer,
            citations=citations,
            route=QueryRoute.RAG,
        )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    samples = root / "samples"
    store = InMemoryDocumentStore()
    vector_store = InMemoryVectorStore()
    embedder = HashEmbedder()
    sparse_encoder = CharSparseEncoder()
    pipeline = IngestionPipeline(
        parser=PyMuPdfParser(image_dir=root / "artifacts" / "images"),
        ocr=None,  # type: ignore[arg-type]
        store=store,
        vector_store=vector_store,
        embedder=embedder,
        sparse_encoder=sparse_encoder,
        cleaner=IdentityCleaner(),
    )

    for pdf in sorted(samples.glob("*.pdf")):
        pipeline.ingest_file(pdf)

    chunks = list(store.chunks.values())
    retrieval = HybridRetrievalService(
        bm25=BM25Retriever(chunks, store.documents),
        dense=DenseRetriever(embedder, vector_store, chunks),
        sparse=SparseRetriever(sparse_encoder, vector_store, chunks),
        reranker=ScoreReranker(),
        rrf_top_k=50,
        reranker_top_k=5,
        score_threshold=0.0,
    )
    generator = ExtractiveAnswerGenerator()

    queries = [
        "个人住房贷款管理办法有哪些要求？",
        "金融机构消费者权益保护怎么评价？",
        "养老理财业务有什么要求？",
    ]
    for query in queries:
        candidates = retrieval.retrieve(query)
        output = generator.generate(query, candidates)
        print("\nQUESTION:", query)
        print("ANSWER:", output.answer)
        print("CITATIONS:", output.citations)


if __name__ == "__main__":
    main()
