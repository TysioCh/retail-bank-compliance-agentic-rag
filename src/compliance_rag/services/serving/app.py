from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from compliance_rag.services.agent.workflow import AgentOrchestrator


class ChatRequest(BaseModel):
    query: str
    selected_doc_ids: list[str] | None = None


def create_app(agent: AgentOrchestrator) -> FastAPI:
    app = FastAPI(title="Compliance Agentic RAG")

    @app.post("/chat")
    def chat(request: ChatRequest) -> dict[str, object]:
        result = agent.run(request.query, request.selected_doc_ids)
        return {
            "answer": result.output.answer,
            "citations": result.output.citations,
            "related_images": result.output.related_images,
            "route": result.output.route.value if result.output.route else None,
            "trace": result.trace,
        }

    @app.get("/summary_candidates")
    def summary_candidates(query: str) -> list[dict[str, str]]:
        return agent.summary_candidates(query)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
