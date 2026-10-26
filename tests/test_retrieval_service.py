import unittest

from compliance_rag.models import DocCandidate
from compliance_rag.services.retrieval.service import HybridRetrievalService


class FakeRetriever:
    def __init__(self, candidates):
        self.candidates = candidates

    def retrieve(self, query, top_k):
        return self.candidates[:top_k]


class FakeReranker:
    def rerank(self, query, candidates):
        return sorted(candidates, key=lambda item: item.score, reverse=True)


class RetrievalServiceTest(unittest.TestCase):
    def test_hybrid_retrieval(self):
        a = DocCandidate("a", "d1", "p1", "text-a", 0.9)
        b = DocCandidate("b", "d2", "p2", "text-b", 0.7)
        c = DocCandidate("c", "d3", "p3", "text-c", 0.5)
        service = HybridRetrievalService(
            bm25=FakeRetriever([a, b]),
            dense=FakeRetriever([b, c]),
            sparse=FakeRetriever([a, c]),
            reranker=FakeReranker(),
            rrf_top_k=3,
            reranker_top_k=2,
            score_threshold=0.4,
        )
        results = service.retrieve("query")
        self.assertEqual([item.chunk_id for item in results], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
