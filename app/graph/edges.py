"""LangGraph conditional edge functions."""
from app.graph.state import AgentState


def should_retrieve_context(state: AgentState) -> str:
    """
    Determine if context retrieval is needed.

    Args:
        state: Current agent state

    Returns:
        "retrieve_context" if needed, "execute_tools" otherwise
    """
    intent = state.get("intent")
    if intent in ["research_query", "temporal_query", "entity_research"]:
        return "retrieve_context"
    return "execute_tools"


def should_use_comparison_tool(state: AgentState) -> str:
    """
    Determine if comparison tool should be used.

    Args:
        state: Current agent state

    Returns:
        "execute_tools" if comparison needed, "generate_response" otherwise
    """
    intent = state.get("intent")
    if intent == "comparison":
        return "execute_tools"
    return "generate_response"


def should_use_trend_tool(state: AgentState) -> str:
    """
    Determine if trend tool should be used.

    Args:
        state: Current agent state

    Returns:
        "execute_tools" if trend analysis needed, "generate_response" otherwise
    """
    intent = state.get("intent")
    if intent == "trend_analysis":
        return "execute_tools"
    return "generate_response"


def should_use_reasoning(state: AgentState) -> str:
    """
    Determine if chain-of-thought reasoning is needed.

    Args:
        state: Current agent state

    Returns:
        "reasoning_step" if reasoning needed, "generate_response" otherwise
    """
    intent = state.get("intent")
    # Use reasoning for complex queries
    complex_intents = ["trend_analysis", "comparison"]
    if intent in complex_intents:
        return "reasoning_step"
    return "generate_response"


def should_generate_response(state: AgentState) -> str:
    """
    Determine if response should be generated.

    Args:
        state: Current agent state

    Returns:
        "generate_response" if ready, "END" otherwise
    """
    # After tools are executed or context is retrieved, generate response
    if state.get("context") or state.get("tools_used"):
        return "generate_response"
    # If no tools needed, generate response directly
    return "generate_response"


def should_handle_error(state: AgentState) -> str:
    """
    Determine if error handling is needed.
    This is used as a conditional edge function from nodes that might error.

    Args:
        state: Current agent state

    Returns:
        "handle_error" if error exists, otherwise the appropriate next node
    """
    if state.get("error"):
        return "handle_error"
    
    # No error - determine next step based on which node we're coming from
    # This is context-dependent, so we check what's available in state
    
    # If coming from retrieve_context: always go to generate_response
    if state.get("context") is not None:
        return "generate_response"
    
    # If coming from execute_tools: go to generate_response if we have results
    if state.get("tools_used"):
        return "generate_response"
    
    # If coming from generate_response: go to END
    if state.get("messages") and any(msg.get("role") == "assistant" for msg in state.get("messages", [])):
        return "END"
    
    # Default fallback
    return "END"

