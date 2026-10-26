import unittest

from compliance_rag.pipelines.eval.metrics import (
    combined_answer_score,
    hit_at_k,
    jaccard_similarity,
    mrr,
    recall_at_k,
)


class MetricsTest(unittest.TestCase):
    def test_jaccard(self):
        self.assertEqual(jaccard_similarity(["a", "b"], ["b", "c"]), 1 / 3)

    def test_recall_and_hit(self):
        self.assertEqual(recall_at_k(["a", "b"], ["a", "c"], 1), 0.5)
        self.assertEqual(hit_at_k(["a"], ["a", "b"], 1), 1.0)

    def test_mrr(self):
        self.assertEqual(mrr([["a"]], [["b", "a"]]), 0.5)

    def test_combined_score(self):
        self.assertEqual(combined_answer_score(0.8, 0.6, 0.5), 0.7)


if __name__ == "__main__":
    unittest.main()
