"""LangGraph state definition."""
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State for the agent graph."""

    messages: Annotated[List[BaseMessage], add_messages]
    """Conversation history as LangChain BaseMessage objects."""

    user_id: str
    """User identifier."""

    conversation_id: str
    """Conversation identifier (used as thread_id for checkpoints)."""

    intent: Optional[str]
    """Classified intent: research_query, comparison, trend_analysis, temporal_query, entity_research, out_of_scope."""

    context: List[Dict[str, Any]]
    """Retrieved documents from RAG or tool results."""

    tools_used: List[str]
    """List of tool names that were called."""

    requires_human: bool
    """Human-in-loop flag (for future use)."""

    error: Optional[str]
    """Error message if any error occurred."""

    sources: List[Dict[str, Any]]
    """Source citations from retrieved documents."""

    usage: Optional[Dict[str, int]]
    """Token usage information from LLM calls."""

    reasoning: Optional[str]
    """Chain-of-thought reasoning for complex queries."""

    validation: Optional[Dict[str, Any]]
    """Response validation results."""

    validation_retried: bool
    """Whether response has been regenerated due to validation issues."""

