from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a source PDF")
    parser.add_argument("--source", required=True)
    parser.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    parser.add_argument("--milvus-uri", default="http://localhost:19530")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    print(
        {
            "source": str(source),
            "mongo_uri": args.mongo_uri,
            "milvus_uri": args.milvus_uri,
        }
    )


if __name__ == "__main__":
    main()
