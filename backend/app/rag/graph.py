"""
LangGraph agent definition.
Defines the StateGraph with conditional routing for different query types.
"""

import logging
from typing import TypedDict
from langgraph.graph import StateGraph, END
from app.rag.nodes import (
    route_query_node,
    rewrite_query_node,
    retrieve_node,
    compress_node,
    generate_node,
    guardrail_node,
    suggestions_node,
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Agent State Definition
# ──────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    """State that flows through the LangGraph agent."""
    # Input
    query: str
    video_id: str
    video_title: str
    session_id: str
    chat_history: list[dict]

    # Processing
    query_type: str  # search | summarize | clarify | contextualize
    domain: str  # educational | tutorial | entertainment | podcast | news | other
    rewritten_queries: list[str]
    retrieved_docs: list
    compressed_docs: list

    # Output
    answer: str
    citations: list
    suggested_questions: list[str]
    guardrail_passed: bool


# ──────────────────────────────────────────────
# Routing Logic
# ──────────────────────────────────────────────

def _route_after_classification(state: AgentState) -> str:
    """Determine the next node based on query type."""
    query_type = state.get("query_type", "search")

    if query_type == "summarize":
        # Summarize needs broad retrieval without heavy rewriting
        return "retrieve"
    elif query_type in ("search", "clarify", "contextualize"):
        # These benefit from query rewriting
        return "rewrite"
    else:
        return "rewrite"


def _route_after_guardrail(state: AgentState) -> str:
    """Route after guard rail check."""
    if state.get("guardrail_passed", True):
        return "suggestions"
    else:
        return "suggestions"  # Still generate suggestions even on failure


# ──────────────────────────────────────────────
# Graph Builder
# ──────────────────────────────────────────────

def build_rag_graph() -> StateGraph:
    """
    Build and compile the RAG agent graph.

    Flow:
    START → route → [rewrite → retrieve | retrieve] → compress → generate → guardrail → suggestions → END
    """
    graph = StateGraph(AgentState)

    # ── Add nodes ──
    graph.add_node("route", route_query_node)
    graph.add_node("rewrite", rewrite_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("compress", compress_node)
    graph.add_node("generate", generate_node)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("suggestions", suggestions_node)

    # ── Entry point ──
    graph.set_entry_point("route")

    # ── Conditional routing after classification ──
    graph.add_conditional_edges(
        "route",
        _route_after_classification,
        {
            "rewrite": "rewrite",
            "retrieve": "retrieve",
        },
    )

    # ── Linear flow: rewrite → retrieve ──
    graph.add_edge("rewrite", "retrieve")

    # ── Linear flow: retrieve → compress → generate → guardrail → suggestions → END ──
    graph.add_edge("retrieve", "compress")
    graph.add_edge("compress", "generate")
    graph.add_edge("generate", "guardrail")

    graph.add_conditional_edges(
        "guardrail",
        _route_after_guardrail,
        {
            "suggestions": "suggestions",
        },
    )

    graph.add_edge("suggestions", END)

    # ── Compile ──
    compiled = graph.compile()
    logger.info("RAG agent graph compiled successfully")

    return compiled


# ──────────────────────────────────────────────
# Singleton Graph Instance
# ──────────────────────────────────────────────

_graph = None


def get_rag_graph():
    """Get or create the singleton RAG graph."""
    global _graph
    if _graph is None:
        _graph = build_rag_graph()
    return _graph


async def run_agent(
    query: str,
    video_id: str,
    video_title: str = "",
    session_id: str = "",
    chat_history: list[dict] = None,
) -> dict:
    """
    Run the RAG agent with the given query.

    Returns a dict with:
    - answer: str
    - citations: list[Citation]
    - suggested_questions: list[str]
    - query_type: str
    """
    graph = get_rag_graph()

    initial_state: AgentState = {
        "query": query,
        "video_id": video_id,
        "video_title": video_title,
        "session_id": session_id,
        "chat_history": chat_history or [],
        "query_type": "search",
        "domain": "other",
        "rewritten_queries": [],
        "retrieved_docs": [],
        "compressed_docs": [],
        "answer": "",
        "citations": [],
        "suggested_questions": [],
        "guardrail_passed": True,
    }

    try:
        # Run the graph
        result = await graph.ainvoke(initial_state)

        return {
            "answer": result.get("answer", "I encountered an issue processing your request."),
            "citations": result.get("citations", []),
            "suggested_questions": result.get("suggested_questions", []),
            "query_type": result.get("query_type", "search"),
            "domain": result.get("domain", "other"),
        }

    except Exception as e:
        logger.error(f"Agent execution failed: {e}", exc_info=True)
        # Return the actual error message temporarily to help with debugging
        return {
            "answer": f"I'm sorry, I encountered an error: {str(e)}. Please check your backend logs for more details.",
            "citations": [],
            "suggested_questions": ["Check API Key", "Check Qdrant Connection"],
            "query_type": "search",
            "domain": "other",
        }
