import unittest

from compliance_rag.core.retrieval import (
    dedupe_candidates,
    fuse_rankings,
    rrf_score,
    top_k_by_score,
)
from compliance_rag.models import DocCandidate


class RetrievalCoreTest(unittest.TestCase):
    def test_rrf_score(self):
        self.assertAlmostEqual(rrf_score([1, 3], k=60), 1 / 61 + 1 / 63)

    def test_fuse_rankings(self):
        scores = fuse_rankings([["a", "b"], ["b", "a"]], k=60)
        self.assertEqual(set(scores), {"a", "b"})
        self.assertAlmostEqual(scores["a"], 1 / 61 + 1 / 62)

    def test_top_k_by_score(self):
        self.assertEqual(top_k_by_score({"a": 1.0, "b": 0.5, "c": 0.9}, 2), [("a", 1.0), ("c", 0.9)])

    def test_dedupe_candidates(self):
        candidates = [
            DocCandidate("1", "d1", "p1", "a", 0.9, {"chunk_id": "1"}),
            DocCandidate("2", "d2", "p2", "b", 0.8, {"chunk_id": "1"}),
            DocCandidate("3", "d3", "p3", "c", 0.7, {"chunk_id": "3"}),
        ]
        deduped = dedupe_candidates(candidates)
        self.assertEqual([item.chunk_id for item in deduped], ["1", "3"])


if __name__ == "__main__":
    unittest.main()
