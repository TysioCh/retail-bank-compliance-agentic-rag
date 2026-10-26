from __future__ import annotations

import argparse
from pathlib import Path

from compliance_rag.adapters.mongo import MongoDocumentStore
from compliance_rag.pipelines.ingestion import IngestionPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    parser.add_argument("--database", default="compliance_rag")
    args = parser.parse_args()

    store = MongoDocumentStore(args.mongo_uri, args.database)
    # The remaining adapters are wired by the runtime dependency container in a
    # real deployment. This script keeps the execution entry explicit.
    print("ingestion entry initialized", store.database.name)
    print("source", Path(args.source).resolve())


if __name__ == "__main__":
    main()
