from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from compliance_rag.models import AgentOutput, QueryRoute

NO_ANSWER = "无答案"


class Guardrail(Protocol):
    def check(self, query: str) -> tuple[bool, str | None]:
        ...


class Router(Protocol):
    def route(self, query: str) -> QueryRoute:
        ...


class RagResponder(Protocol):
    def respond(self, query: str) -> AgentOutput:
        ...


class RecoveryResponder(Protocol):
    def rewrite_query(self, query: str) -> str:
        ...

    def hyde_query(self, query: str) -> str:
        ...


class SummaryResponder(Protocol):
    def list_documents(self, query: str) -> list[dict[str, str]]:
        ...

    def summarize(self, query: str, selected_doc_ids: list[str]) -> AgentOutput:
        ...


class WebSearchResponder(Protocol):
    def respond(self, query: str) -> AgentOutput:
        ...


class Postprocessor(Protocol):
    def process(self, output: AgentOutput) -> AgentOutput:
        ...


@dataclass
class AgentRunResult:
    output: AgentOutput
    trace: dict[str, Any] = field(default_factory=dict)


class AgentOrchestrator:
    def __init__(
        self,
        guardrail: Guardrail,
        router: Router,
        rag: RagResponder,
        recovery: RecoveryResponder,
        summary: SummaryResponder,
        web_search: WebSearchResponder,
        postprocessor: Postprocessor,
        query_rewrite_enabled: bool = True,
        hyde_enabled: bool = True,
    ) -> None:
        self.guardrail = guardrail
        self.router = router
        self.rag = rag
        self.recovery = recovery
        self.summary = summary
        self.web_search = web_search
        self.postprocessor = postprocessor
        self.query_rewrite_enabled = query_rewrite_enabled
        self.hyde_enabled = hyde_enabled

    def run(
        self,
        query: str,
        selected_doc_ids: list[str] | None = None,
    ) -> AgentRunResult:
        allowed, reason = self.guardrail.check(query)
        if not allowed:
            output = AgentOutput(
                answer=NO_ANSWER,
                route=QueryRoute.REJECT,
                trace={"reject_reason": reason},
            )
            return AgentRunResult(output=self.postprocessor.process(output), trace=output.trace)

        route = self.router.route(query)
        if route == QueryRoute.SUMMARY:
            output = self.summary.summarize(query, selected_doc_ids or [])
            output.route = QueryRoute.SUMMARY
            output = self.postprocessor.process(output)
            return AgentRunResult(output=output, trace=output.trace)

        if route == QueryRoute.WEB_SEARCH:
            output = self.web_search.respond(query)
            output.route = QueryRoute.WEB_SEARCH
            output = self.postprocessor.process(output)
            return AgentRunResult(output=output, trace=output.trace)

        if route == QueryRoute.REJECT:
            output = AgentOutput(answer=NO_ANSWER, route=QueryRoute.REJECT)
            output = self.postprocessor.process(output)
            return AgentRunResult(output=output, trace=output.trace)

        return self._run_rag_flow(query)

    def summary_candidates(self, query: str) -> list[dict[str, str]]:
        return self.summary.list_documents(query)

    def _run_rag_flow(self, query: str) -> AgentRunResult:
        attempts = [("initial", query)]
        output = self.rag.respond(query)
        if output.answer != NO_ANSWER:
            output.route = QueryRoute.RAG
            output = self.postprocessor.process(output)
            return AgentRunResult(output=output, trace={"attempts": attempts})

        if self.query_rewrite_enabled:
            rewritten = self.recovery.rewrite_query(query)
            attempts.append(("rewrite", rewritten))
            output = self.rag.respond(rewritten)
            if output.answer != NO_ANSWER:
                output.route = QueryRoute.RAG
                output = self.postprocessor.process(output)
                return AgentRunResult(output=output, trace={"attempts": attempts})

        if self.hyde_enabled:
            hyde_query = self.recovery.hyde_query(query)
            attempts.append(("hyde", hyde_query))
            output = self.rag.respond(hyde_query)
            if output.answer != NO_ANSWER:
                output.route = QueryRoute.RAG
                output = self.postprocessor.process(output)
                return AgentRunResult(output=output, trace={"attempts": attempts})

        output = AgentOutput(answer=NO_ANSWER, route=QueryRoute.RAG)
        output = self.postprocessor.process(output)
        return AgentRunResult(output=output, trace={"attempts": attempts})
