"""Tests for LangGraph graph."""
import pytest
from unittest.mock import AsyncMock, patch
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
    should_generate_response,
    should_handle_error,
)


@pytest.mark.asyncio
async def test_classify_intent(mock_llm_client, sample_agent_state):
    """Test intent classification node."""
    with patch("app.graph.nodes._llm_client", mock_llm_client):
        state = await classify_intent(sample_agent_state)
        assert state["intent"] == "research_query"
        mock_llm_client.classify_intent.assert_called_once()


@pytest.mark.asyncio
async def test_retrieve_context(mock_rag_client, sample_agent_state):
    """Test context retrieval node."""
    with patch("app.graph.nodes._rag_client", mock_rag_client):
        state = await retrieve_context(sample_agent_state)
        assert len(state["context"]) > 0
        assert len(state["sources"]) > 0
        mock_rag_client.query.assert_called_once()


@pytest.mark.asyncio
async def test_execute_tools(mock_llm_client, sample_agent_state):
    """Test tool execution node."""
    sample_agent_state["intent"] = "research_query"
    with patch("app.graph.nodes.rag_search") as mock_rag_tool:
        mock_rag_tool.ainvoke = AsyncMock(return_value='{"results": [], "count": 0}')
        state = await execute_tools(sample_agent_state)
        assert "rag_search" in state["tools_used"]


@pytest.mark.asyncio
async def test_generate_response(mock_llm_client, sample_agent_state):
    """Test response generation node."""
    sample_agent_state["context"] = [{"content": "Test context"}]
    with patch("app.graph.nodes._llm_client", mock_llm_client):
        state = await generate_response(sample_agent_state)
        assert len(state["messages"]) > 1
        assert state["messages"][-1]["role"] == "assistant"
        assert state["usage"] is not None


@pytest.mark.asyncio
async def test_handle_error(sample_agent_state):
    """Test error handling node."""
    sample_agent_state["error"] = "Test error"
    state = await handle_error(sample_agent_state)
    assert len(state["messages"]) > 0
    assert state["messages"][-1]["role"] == "assistant"
    assert "error" in state["messages"][-1]["content"].lower()
    assert state["error"] is None


def test_should_retrieve_context(sample_agent_state):
    """Test conditional edge for context retrieval."""
    sample_agent_state["intent"] = "research_query"
    result = should_retrieve_context(sample_agent_state)
    assert result == "retrieve_context"

    sample_agent_state["intent"] = "comparison"
    result = should_retrieve_context(sample_agent_state)
    assert result == "execute_tools"


def test_should_generate_response(sample_agent_state):
    """Test conditional edge for response generation."""
    sample_agent_state["context"] = [{"content": "Test"}]
    result = should_generate_response(sample_agent_state)
    assert result == "generate_response"


def test_should_handle_error(sample_agent_state):
    """Test conditional edge for error handling."""
    sample_agent_state["error"] = "Test error"
    result = should_handle_error(sample_agent_state)
    assert result == "handle_error"

    sample_agent_state["error"] = None
    result = should_handle_error(sample_agent_state)
    assert result == "END"

