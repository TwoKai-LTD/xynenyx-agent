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

    # Add conditional edges from classify_intent (with error handling)
    def route_from_classify_intent(state: AgentState) -> str:
        """Route from classify_intent, checking for errors first."""
        if state.get("error"):
            return "handle_error"
        result = should_retrieve_context(state)
        # Ensure result is in the mapping
        if result not in ["retrieve_context", "execute_tools", "handle_error"]:
            logger.warning(f"Unexpected route from classify_intent: {result}, defaulting to retrieve_context")
            return "retrieve_context"
        return result
    
    graph.add_conditional_edges(
        "classify_intent",
        route_from_classify_intent,
        {
            "retrieve_context": "retrieve_context",
            "execute_tools": "execute_tools",
            "handle_error": "handle_error",
        },
    )

    # Add conditional edge from retrieve_context (with error handling)
    def route_from_retrieve_context(state: AgentState) -> str:
        """Route from retrieve_context, checking for errors first."""
        if state.get("error"):
            return "handle_error"
        return "generate_response"
    
    graph.add_conditional_edges(
        "retrieve_context",
        route_from_retrieve_context,
        {
            "generate_response": "generate_response",
            "handle_error": "handle_error",
        },
    )

    # Add conditional edges from execute_tools (with error handling)
    def route_from_execute_tools(state: AgentState) -> str:
        """Route from execute_tools, checking for errors first."""
        if state.get("error"):
            return "handle_error"
        return should_generate_response(state)
    
    graph.add_conditional_edges(
        "execute_tools",
        route_from_execute_tools,
        {
            "generate_response": "generate_response",
            "handle_error": "handle_error",
        },
    )

    # Add conditional edge from generate_response (with error handling)
    def route_from_generate_response(state: AgentState) -> str:
        """Route from generate_response, checking for errors first."""
        if state.get("error"):
            return "handle_error"
        return "END"
    
    graph.add_conditional_edges(
        "generate_response",
        route_from_generate_response,
        {
            "END": END,
            "handle_error": "handle_error",
        },
    )

    # Add edge from handle_error to END
    graph.add_edge("handle_error", END)

    # Compile graph with checkpointer if provided
    # LangGraph expects checkpointer to be passed as a dict in the config
    if checkpointer:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()


# Global graph instance (initialized in main.py)
_agent_graph = None


def get_agent_graph():
    """Get or create the agent graph instance."""
    global _agent_graph
    if _agent_graph is None:
        # Temporarily disable checkpointer until we implement proper LangGraph checkpointer interface
        # checkpointer = SupabaseCheckpointer() if settings.checkpoint_enabled else None
        checkpointer = None  # Disabled for now
        _agent_graph = build_agent_graph(checkpointer=checkpointer)
    return _agent_graph

