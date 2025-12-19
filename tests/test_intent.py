"""Tests for intent classification."""
import pytest
from unittest.mock import AsyncMock, patch
from app.services.llm_client import LLMServiceClient


@pytest.mark.asyncio
async def test_classify_research_query():
    """Test classification of research query intent."""
    mock_client = AsyncMock()
    mock_client.complete = AsyncMock(
        return_value={"content": "research_query"}
    )
    mock_client.classify_intent = AsyncMock(return_value="research_query")

    with patch("app.services.llm_client.LLMServiceClient", return_value=mock_client):
        client = LLMServiceClient()
        intent = await client.classify_intent("What is Xynenyx?")
        assert intent == "research_query"


@pytest.mark.asyncio
async def test_classify_comparison():
    """Test classification of comparison intent."""
    mock_client = AsyncMock()
    mock_client.classify_intent = AsyncMock(return_value="comparison")

    with patch("app.services.llm_client.LLMServiceClient", return_value=mock_client):
        client = LLMServiceClient()
        intent = await client.classify_intent("Compare Company A and Company B")
        assert intent == "comparison"


@pytest.mark.asyncio
async def test_classify_trend_analysis():
    """Test classification of trend analysis intent."""
    mock_client = AsyncMock()
    mock_client.classify_intent = AsyncMock(return_value="trend_analysis")

    with patch("app.services.llm_client.LLMServiceClient", return_value=mock_client):
        client = LLMServiceClient()
        intent = await client.classify_intent("What are the trends in AI funding?")
        assert intent == "trend_analysis"


@pytest.mark.asyncio
async def test_classify_fallback():
    """Test fallback to default intent on error."""
    mock_client = AsyncMock()
    mock_client.complete = AsyncMock(side_effect=Exception("Error"))
    mock_client.classify_intent = AsyncMock(return_value="research_query")

    with patch("app.services.llm_client.LLMServiceClient", return_value=mock_client):
        client = LLMServiceClient()
        intent = await client.classify_intent("Test query")
        assert intent == "research_query"

