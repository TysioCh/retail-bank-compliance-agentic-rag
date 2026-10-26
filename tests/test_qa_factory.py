import unittest

from compliance_rag.pipelines.qa_factory import QADatasetFactory, QAPair, expand_queries


class QAFactoryTest(unittest.TestCase):
    def test_expand_queries(self):
        qa = QAPair(unique_id="q1", question="原始问题", answer="答案")
        expanded = expand_queries(qa, ["表达1", "表达2"])
        self.assertEqual(len(expanded), 2)
        self.assertEqual(expanded[0].unique_id, "q1:v0")

    def test_dataset_factory(self):
        initial = [QAPair("q1", "问题1", "答案"), QAPair("q2", "问题2", "无答案")]
        negatives = [QAPair("n1", "闲聊", "无答案", is_negative=True)]
        factory = QADatasetFactory(
            filter_fn=lambda qa: qa.answer != "无答案",
            query_expander=lambda qa: [qa.question + " A", qa.question + " B"],
        )
        result = factory.build(initial, negatives, train_ratio=0.9)
        self.assertEqual(len(result["train"]), 2)
        self.assertEqual(len(result["test"]), 1)


if __name__ == "__main__":
    unittest.main()
