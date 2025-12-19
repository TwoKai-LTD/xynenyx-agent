"""Pytest configuration and fixtures."""
import pytest
from unittest.mock import AsyncMock, Mock, MagicMock, patch
from typing import Dict, Any
from app.graph.state import AgentState


@pytest.fixture
def mock_llm_client():
    """Mock LLM service client."""
    mock = AsyncMock()
    mock.complete = AsyncMock(
        return_value={
            "content": "Test response",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
    )
    mock.complete_stream = AsyncMock()
    mock.classify_intent = AsyncMock(return_value="research_query")
    return mock


@pytest.fixture
def mock_rag_client():
    """Mock RAG service client."""
    mock = AsyncMock()
    mock.query = AsyncMock(
        return_value={
            "query": "test query",
            "results": [
                {
                    "content": "Test content",
                    "metadata": {"title": "Test"},
                    "document_id": "doc-1",
                    "chunk_id": "chunk-1",
                    "similarity": 0.9,
                }
            ],
            "count": 1,
            "search_mode": "hybrid",
        }
    )
    return mock


@pytest.fixture
def mock_supabase_client():
    """Mock Supabase client."""
    mock = AsyncMock()
    mock.get_conversation = AsyncMock(
        return_value={
            "id": "conv-1",
            "user_id": "user-1",
            "title": "Test Conversation",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        }
    )
    mock.get_messages = AsyncMock(return_value=[])
    mock.save_message = AsyncMock(
        return_value={
            "id": "msg-1",
            "conversation_id": "conv-1",
            "role": "user",
            "content": "Test message",
        }
    )
    mock.create_conversation = AsyncMock(
        return_value={
            "id": "conv-1",
            "user_id": "user-1",
            "title": "New Conversation",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
        }
    )
    mock.list_conversations = AsyncMock(return_value=[])
    mock.delete_conversation = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def sample_agent_state() -> AgentState:
    """Sample agent state for testing."""
    return {
        "messages": [
            {"role": "user", "content": "What is Xynenyx?"}
        ],
        "user_id": "user-1",
        "conversation_id": "conv-1",
        "intent": None,
        "context": [],
        "tools_used": [],
        "requires_human": False,
        "error": None,
        "sources": [],
        "usage": None,
    }
