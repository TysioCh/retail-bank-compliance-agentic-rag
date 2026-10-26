from __future__ import annotations

import math
from pathlib import Path

from rank_bm25 import BM25Okapi

from compliance_rag.adapters.in_memory import (
    CharSparseEncoder,
    HashEmbedder,
    IdentityCleaner,
    InMemoryDocumentStore,
    InMemoryVectorStore,
)
from compliance_rag.adapters.pdf_parser import PyMuPdfParser
from compliance_rag.models import DocCandidate
from compliance_rag.pipelines.ingestion import IngestionPipeline
from compliance_rag.services.retrieval.service import HybridRetrievalService


def tokenize(text: str) -> list[str]:
    return [char for char in text if not char.isspace()]


class BM25Retriever:
    def __init__(self, chunks, documents) -> None:
        self.chunks = chunks
        self.documents = documents
        self.corpus = [tokenize(chunk["text"]) for chunk in chunks]
        self.index = BM25Okapi(self.corpus)

    def retrieve(self, query: str, top_k: int):
        scores = self.index.get_scores(tokenize(query))
        ranked = sorted(range(len(self.chunks)), key=lambda index: scores[index], reverse=True)[:top_k]
        return [self._candidate(index, scores[index]) for index in ranked]

    def _candidate(self, index, score):
        chunk = self.chunks[index]
        return DocCandidate(
            chunk_id=chunk["chunk_id"],
            doc_id=chunk["parent_id"],
            parent_id=chunk["parent_id"],
            text=chunk["text"],
            score=float(score),
        )


class DenseRetriever:
    def __init__(self, embedder, vector_store, chunks) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}

    def retrieve(self, query: str, top_k: int):
        vector = self.embedder.embed_texts([query])[0]
        results = self.vector_store.search(vector, top_k)
        return [
            DocCandidate(
                chunk_id=item["chunk_id"],
                doc_id=self.chunks_by_id[item["chunk_id"]]["parent_id"],
                parent_id=self.chunks_by_id[item["chunk_id"]]["parent_id"],
                text=self.chunks_by_id[item["chunk_id"]]["text"],
                score=float(item["score"]),
            )
            for item in results
        ]


class SparseRetriever:
    def __init__(self, encoder, vector_store, chunks) -> None:
        self.encoder = encoder
        self.vector_store = vector_store
        self.chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}

    def retrieve(self, query: str, top_k: int):
        query_vector = self.encoder.encode([query])[0]
        scored = []
        for chunk_id, vector in self.vector_store.sparse.items():
            score = sum(query_vector.get(index, 0.0) * value for index, value in vector.items())
            scored.append((chunk_id, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            DocCandidate(
                chunk_id=chunk_id,
                doc_id=self.chunks_by_id[chunk_id]["parent_id"],
                parent_id=self.chunks_by_id[chunk_id]["parent_id"],
                text=self.chunks_by_id[chunk_id]["text"],
                score=float(score),
            )
            for chunk_id, score in scored[:top_k]
        ]


class ScoreReranker:
    def rerank(self, query: str, candidates: list[DocCandidate]):
        return sorted(candidates, key=lambda item: item.score, reverse=True)


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
    service = HybridRetrievalService(
        bm25=BM25Retriever(chunks, store.documents),
        dense=DenseRetriever(embedder, vector_store, chunks),
        sparse=SparseRetriever(sparse_encoder, vector_store, chunks),
        reranker=ScoreReranker(),
        rrf_top_k=50,
        reranker_top_k=5,
        score_threshold=0.0,
    )

    for query in [
        "个人住房贷款管理办法有哪些要求？",
        "金融机构消费者权益保护怎么评价？",
        "养老理财业务有什么要求？",
    ]:
        results = service.retrieve(query)
        print("\nQUERY:", query)
        for item in results:
            print("-", item.chunk_id, round(item.score, 4), item.text[:70].replace("\n", " "))


if __name__ == "__main__":
    main()
