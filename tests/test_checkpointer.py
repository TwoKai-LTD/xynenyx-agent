"""Tests for Supabase checkpointer."""
import pytest
from unittest.mock import AsyncMock, Mock, patch
from app.graph.checkpointer import SupabaseCheckpointer


@pytest.fixture
def mock_supabase():
    """Mock Supabase client."""
    mock = Mock()
    mock.table = Mock(return_value=mock)
    mock.upsert = Mock(return_value=mock)
    mock.select = Mock(return_value=mock)
    mock.eq = Mock(return_value=mock)
    mock.order = Mock(return_value=mock)
    mock.limit = Mock(return_value=mock)
    mock.delete = Mock(return_value=mock)
    mock.lt = Mock(return_value=mock)
    mock.execute = Mock(return_value=Mock(data=[{"checkpoint_id": "cp-1", "checkpoint": {}}]))
    return mock


@pytest.mark.asyncio
async def test_put_checkpoint(mock_supabase):
    """Test saving a checkpoint."""
    with patch("app.graph.checkpointer.create_client", return_value=mock_supabase):
        checkpointer = SupabaseCheckpointer()
        await checkpointer.put(
            thread_id="thread-1",
            checkpoint_id="cp-1",
            checkpoint={"state": "test"},
        )
        mock_supabase.table.assert_called_with("agent_checkpoints")
        mock_supabase.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_get_checkpoint(mock_supabase):
    """Test retrieving a checkpoint."""
    mock_supabase.execute = Mock(
        return_value=Mock(
            data=[{
                "checkpoint_id": "cp-1",
                "parent_checkpoint_id": None,
                "checkpoint": {"state": "test"},
                "metadata": {},
            }]
        )
    )
    with patch("app.graph.checkpointer.create_client", return_value=mock_supabase):
        checkpointer = SupabaseCheckpointer()
        result = await checkpointer.get("thread-1", "cp-1")
        assert result is not None
        assert result["checkpoint_id"] == "cp-1"


@pytest.mark.asyncio
async def test_list_checkpoints(mock_supabase):
    """Test listing checkpoints."""
    mock_supabase.execute = Mock(
        return_value=Mock(
            data=[
                {
                    "checkpoint_id": "cp-1",
                    "parent_checkpoint_id": None,
                    "checkpoint": {"state": "test"},
                    "metadata": {},
                    "created_at": "2024-01-01T00:00:00",
                }
            ]
        )
    )
    with patch("app.graph.checkpointer.create_client", return_value=mock_supabase):
        checkpointer = SupabaseCheckpointer()
        result = await checkpointer.list("thread-1")
        assert len(result) > 0


@pytest.mark.asyncio
async def test_delete_checkpoint(mock_supabase):
    """Test deleting a checkpoint."""
    with patch("app.graph.checkpointer.create_client", return_value=mock_supabase):
        checkpointer = SupabaseCheckpointer()
        await checkpointer.delete("thread-1", "cp-1")
        mock_supabase.delete.assert_called()

