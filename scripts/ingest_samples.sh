#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAMPLES_DIR="$ROOT_DIR/samples"

for pdf in "$SAMPLES_DIR"/*.pdf; do
  echo "ingesting: $pdf"
  python -m compliance_rag.cli.ingest --source "$pdf"
done
