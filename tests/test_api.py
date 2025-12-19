"""Tests for API endpoints."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.schemas.requests import ChatRequest, ConversationCreateRequest


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_graph():
    """Mock agent graph."""
    mock = AsyncMock()
    mock.ainvoke = AsyncMock(
        return_value={
            "messages": [
                {"role": "user", "content": "Test message"},
                {"role": "assistant", "content": "Test response"},
            ],
            "sources": [],
            "tools_used": [],
            "usage": {"total_tokens": 30},
        }
    )
    return mock


@pytest.mark.asyncio
async def test_chat_endpoint(client, mock_supabase_client, mock_graph):
    """Test POST /chat endpoint."""
    with patch("app.routers.chat.supabase_client", mock_supabase_client), \
         patch("app.routers.chat.get_agent_graph", return_value=mock_graph):
        response = client.post(
            "/chat",
            json={
                "message": "Test message",
                "conversation_id": "conv-1",
                "user_id": "user-1",
                "stream": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["conversation_id"] == "conv-1"


@pytest.mark.asyncio
async def test_chat_stream_endpoint(client, mock_supabase_client, mock_graph):
    """Test POST /chat/stream endpoint."""
    with patch("app.routers.chat.supabase_client", mock_supabase_client), \
         patch("app.routers.chat.get_agent_graph", return_value=mock_graph):
        response = client.post(
            "/chat/stream",
            json={
                "message": "Test message",
                "conversation_id": "conv-1",
                "user_id": "user-1",
                "stream": True,
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"


@pytest.mark.asyncio
async def test_list_conversations(client, mock_supabase_client):
    """Test GET /conversations endpoint."""
    mock_supabase_client.list_conversations = AsyncMock(return_value=[])
    with patch("app.routers.conversations.supabase_client", mock_supabase_client):
        response = client.get(
            "/conversations",
            headers={"X-User-ID": "user-1"},
        )
        assert response.status_code == 200
        assert isinstance(response.json(), list)


@pytest.mark.asyncio
async def test_get_conversation(client, mock_supabase_client):
    """Test GET /conversations/{id} endpoint."""
    with patch("app.routers.conversations.supabase_client", mock_supabase_client):
        response = client.get(
            "/conversations/conv-1",
            headers={"X-User-ID": "user-1"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "conv-1"


@pytest.mark.asyncio
async def test_create_conversation(client, mock_supabase_client):
    """Test POST /conversations endpoint."""
    with patch("app.routers.conversations.supabase_client", mock_supabase_client):
        response = client.post(
            "/conversations",
            json={"title": "Test Conversation"},
            headers={"X-User-ID": "user-1"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "id" in data


@pytest.mark.asyncio
async def test_delete_conversation(client, mock_supabase_client):
    """Test DELETE /conversations/{id} endpoint."""
    with patch("app.routers.conversations.supabase_client", mock_supabase_client):
        response = client.delete(
            "/conversations/conv-1",
            headers={"X-User-ID": "user-1"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "message" in data


@pytest.mark.asyncio
async def test_health_endpoint(client):
    """Test GET /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_ready_endpoint(client):
    """Test GET /ready endpoint."""
    with patch("app.main.get_agent_graph", return_value=MagicMock()):
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"

