from __future__ import annotations

from langgraph.graph import END, StateGraph

from compliance_rag.services.agent.workflow import AgentOrchestrator


def build_langgraph(orchestrator: AgentOrchestrator):
    """Build the LangGraph wrapper around the pure orchestrator.

    The pure orchestrator is intentionally the source of truth for now.
    LangGraph is used to expose the same decision flow as nodes and edges.
    """
    graph = StateGraph(dict)
    graph.add_node("guardrail", lambda state: state)
    graph.add_node("router", lambda state: state)
    graph.add_node("rag", lambda state: state)
    graph.add_node("recovery", lambda state: state)
    graph.add_node("summary", lambda state: state)
    graph.add_node("web_search", lambda state: state)
    graph.add_node("reject", lambda state: state)
    graph.add_node("postprocess", lambda state: state)
    graph.set_entry_point("guardrail")
    graph.add_edge("guardrail", "router")
    graph.add_edge("router", "rag")
    graph.add_edge("router", "summary")
    graph.add_edge("router", "web_search")
    graph.add_edge("router", "reject")
    graph.add_edge("rag", "recovery")
    graph.add_edge("rag", "postprocess")
    graph.add_edge("recovery", "rag")
    graph.add_edge("summary", "postprocess")
    graph.add_edge("web_search", "postprocess")
    graph.add_edge("reject", END)
    graph.add_edge("postprocess", END)
    return graph.compile()
