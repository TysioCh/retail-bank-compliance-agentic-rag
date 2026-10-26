import unittest

from compliance_rag.models import AgentOutput, QueryRoute
from compliance_rag.services.agent.workflow import AgentOrchestrator, NO_ANSWER


class AllowGuardrail:
    def check(self, query):
        return True, None


class FakeRouter:
    def route(self, query):
        return QueryRoute.RAG


class FakeRag:
    def __init__(self, answers):
        self.answers = answers
        self.calls = 0

    def respond(self, query):
        self.calls += 1
        return AgentOutput(answer=self.answers[min(self.calls - 1, len(self.answers) - 1)])


class FakeRecovery:
    def rewrite_query(self, query):
        return query + " rewritten"

    def hyde_query(self, query):
        return query + " hypothetical answer"


class FakeSummary:
    def list_documents(self, query):
        return [{"doc_id": "d1", "title": "政策A"}]

    def summarize(self, query, selected_doc_ids):
        return AgentOutput(answer="总结", citations=["d1"])


class FakeWebSearch:
    def respond(self, query):
        return AgentOutput(answer="联网答案")


class FakePostprocessor:
    def process(self, output):
        output.answer = output.answer.strip()
        return output


class AgentWorkflowTest(unittest.TestCase):
    def test_recovery_order(self):
        rag = FakeRag([NO_ANSWER, NO_ANSWER, "改写后命中"])
        orchestrator = AgentOrchestrator(
            guardrail=AllowGuardrail(),
            router=FakeRouter(),
            rag=rag,
            recovery=FakeRecovery(),
            summary=FakeSummary(),
            web_search=FakeWebSearch(),
            postprocessor=FakePostprocessor(),
        )
        result = orchestrator.run("query")
        self.assertEqual(result.output.answer, "改写后命中")
        self.assertEqual(rag.calls, 3)


if __name__ == "__main__":
    unittest.main()
