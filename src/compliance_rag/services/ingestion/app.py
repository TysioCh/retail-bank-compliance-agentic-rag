from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from compliance_rag.errors import IngestionError


class IngestRequest(BaseModel):
    source_path: str


def create_app(ingestion_pipeline) -> FastAPI:
    app = FastAPI(title="Compliance RAG Ingestion Service")

    @app.post("/ingest")
    def ingest(request: IngestRequest) -> dict[str, object]:
        try:
            return ingestion_pipeline.ingest_file(Path(request.source_path))
        except IngestionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
