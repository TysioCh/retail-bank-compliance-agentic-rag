import tempfile
import unittest
from pathlib import Path

from compliance_rag.adapters.interfaces import ParsedDocument
from compliance_rag.core.chunking import Chunk
from compliance_rag.pipelines.ingestion import IngestionPipeline, compute_document_hash


class FakeParser:
    def parse(self, path: Path) -> ParsedDocument:
        return ParsedDocument(
            source_path=path,
            document_id="doc-1",
            text="section one\nsection two\nsection three",
            metadata={},
        )


class FakeOCR:
    def extract_tables(self, image_path, page):
        return []


class FakeStore:
    def __init__(self):
        self.hashes: set[str] = set()
        self.documents: list[ParsedDocument] = []
        self.chunks: list[Chunk] = []
        self.statuses: dict[str, str] = {}

    def document_exists_by_hash(self, document_hash: str) -> bool:
        return document_hash in self.hashes

    def upsert_document(self, document_id, document, document_hash):
        self.hashes.add(document_hash)
        self.documents.append(document)

    def upsert_chunks(self, chunks):
        self.chunks.extend(chunks)

    def mark_document_status(self, document_id, status, error=None):
        self.statuses[document_id] = status


class FakeVectorStore:
    def __init__(self):
        self.calls = 0

    def upsert_chunk_embeddings(self, chunks, dense_vectors, sparse_vectors):
        self.calls += 1


class FakeEmbedder:
    def embed_texts(self, texts):
        return [[0.1] for _ in texts]


class FakeSparseEncoder:
    def encode(self, texts):
        return [{0: 1.0} for _ in texts]


class FakeCleaner:
    def clean_to_markdown(self, text):
        return "# cleaned\n" + text


class IngestionTest(unittest.TestCase):
    def test_compute_document_hash(self):
        self.assertEqual(len(compute_document_hash(b"hello")), 64)
        self.assertNotEqual(compute_document_hash(b"hello"), compute_document_hash(b"world"))

    def test_ingestion_is_idempotent(self):
        store = FakeStore()
        pipeline = IngestionPipeline(
            parser=FakeParser(),
            ocr=FakeOCR(),
            store=store,
            vector_store=FakeVectorStore(),
            embedder=FakeEmbedder(),
            sparse_encoder=FakeSparseEncoder(),
            cleaner=FakeCleaner(),
        )
        with tempfile.NamedTemporaryFile() as tmp:
            path = Path(tmp.name)
            path.write_bytes(b"same-content")
            first = pipeline.ingest_file(path)
            second = pipeline.ingest_file(path)

        self.assertFalse(first["deduplicated"])
        self.assertTrue(second["deduplicated"])
        self.assertEqual(len(store.documents), 1)


if __name__ == "__main__":
    unittest.main()
