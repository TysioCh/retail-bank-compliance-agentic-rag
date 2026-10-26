import unittest

from compliance_rag.core.chunking import split_parent_child


class ChunkingTest(unittest.TestCase):
    def test_split_parent_child(self):
        text = "abcdefghij" * 20
        chunks = split_parent_child(
            text,
            parent_id="p1",
            child_token_limit=50,
            child_overlap_tokens=10,
        )
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.parent_id == "p1" for chunk in chunks))
        self.assertTrue(all(chunk.chunk_id.startswith("p1:child:") for chunk in chunks))

    def test_empty_text(self):
        self.assertEqual(split_parent_child("", parent_id="p1"), [])


if __name__ == "__main__":
    unittest.main()
