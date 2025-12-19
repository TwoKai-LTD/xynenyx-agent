"""LangGraph graph construction."""
import logging
from langgraph.graph import StateGraph, END
from app.graph.state import AgentState
from app.graph.nodes import (
    classify_intent,
    retrieve_context,
    execute_tools,
    generate_response,
    handle_error,
)
from app.graph.edges import (
    should_retrieve_context,
    should_use_comparison_tool,
    should_use_trend_tool,
    should_generate_response,
    should_handle_error,
)
from app.graph.checkpointer import SupabaseCheckpointer
from app.config import settings

logger = logging.getLogger(__name__)


def build_agent_graph(checkpointer: SupabaseCheckpointer | None = None):
    """
    Build the LangGraph agent graph.

    Args:
        checkpointer: Optional checkpointer for state persistence

    Returns:
        Compiled LangGraph graph
    """
    # Create state graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("execute_tools", execute_tools)
    graph.add_node("generate_response", generate_response)
    graph.add_node("handle_error", handle_error)

    # Set entry point
    graph.set_entry_point("classify_intent")

    # Add conditional edges from classify_intent
    graph.add_conditional_edges(
        "classify_intent",
        should_retrieve_context,
        {
            "retrieve_context": "retrieve_context",
            "execute_tools": "execute_tools",
        },
    )

    # Add edge from retrieve_context to generate_response
    graph.add_edge("retrieve_context", "generate_response")

    # Add conditional edges from execute_tools
    graph.add_conditional_edges(
        "execute_tools",
        should_generate_response,
        {
            "generate_response": "generate_response",
        },
    )

    # Add edge from generate_response to END
    graph.add_edge("generate_response", END)

    # Add conditional edge for error handling (from any node that might error)
    # Note: In practice, errors are handled within nodes, but we can add explicit error handling
    graph.add_conditional_edges(
        "handle_error",
        should_handle_error,
        {
            "END": END,
        },
    )

    # Compile graph with checkpointer if provided
    if checkpointer:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()


# Global graph instance (initialized in main.py)
_agent_graph = None


def get_agent_graph():
    """Get or create the agent graph instance."""
    global _agent_graph
    if _agent_graph is None:
        checkpointer = SupabaseCheckpointer() if settings.checkpoint_enabled else None
        _agent_graph = build_agent_graph(checkpointer=checkpointer)
    return _agent_graph

