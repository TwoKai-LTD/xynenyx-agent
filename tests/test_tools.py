"""Tests for agent tools."""
import pytest
from unittest.mock import AsyncMock, patch
from app.tools.rag_tool import rag_search
from app.tools.comparison_tool import compare_entities
from app.tools.trend_tool import analyze_trends
from app.tools.calculator import calculate


@pytest.mark.asyncio
async def test_rag_search(mock_rag_client):
    """Test RAG search tool."""
    with patch("app.tools.rag_tool._rag_client", mock_rag_client):
        result = await rag_search.ainvoke({
            "query": "test query",
            "top_k": 5,
        })
        assert "results" in result
        assert "count" in result
        mock_rag_client.query.assert_called_once()


@pytest.mark.asyncio
async def test_compare_entities(mock_rag_client):
    """Test comparison tool."""
    mock_rag_client.query = AsyncMock(
        return_value={
            "results": [
                {
                    "content": "Company A raised $10M",
                    "metadata": {"funding_amount": 10.0, "funding_round": "Series A"},
                }
            ],
            "count": 1,
        }
    )
    with patch("app.tools.comparison_tool._rag_client", mock_rag_client):
        result = await compare_entities.ainvoke({
            "entities": ["Company A", "Company B"],
            "query_context": "funding",
        })
        assert "Company A" in result or "error" in result


@pytest.mark.asyncio
async def test_analyze_trends(mock_rag_client):
    """Test trend analysis tool."""
    mock_rag_client.query = AsyncMock(
        return_value={
            "results": [
                {
                    "content": "AI startup raised funding",
                    "metadata": {"sector": "AI", "funding_amount": 5.0},
                }
            ],
            "count": 1,
        }
    )
    with patch("app.tools.trend_tool._rag_client", mock_rag_client):
        result = await analyze_trends.ainvoke({
            "query": "AI funding trends",
            "time_period": None,
        })
        assert "total_deals" in result or "error" in result


def test_calculator():
    """Test calculator tool."""
    result = calculate.invoke({"expression": "2 + 2"})
    assert result == "4"

    result = calculate.invoke({"expression": "10 * 5"})
    assert result == "50"

    result = calculate.invoke({"expression": "20% of 100"})
    assert result == "20.0"

    # Test error handling
    result = calculate.invoke({"expression": "invalid expression"})
    assert "Error" in result

