"""RAG search tool for LangGraph."""
import json
from langchain_core.tools import tool
from typing import List, Dict, Any, Optional
from app.services.rag_client import RAGServiceClient

_rag_client = RAGServiceClient()


@tool
async def rag_search(
    query: str,
    top_k: int = 10,
    date_filter: Optional[str] = None,
    company_filter: Optional[List[str]] = None,
    investor_filter: Optional[List[str]] = None,
    sector_filter: Optional[List[str]] = None,
) -> str:
    """
    Search the knowledge base for information about startups, funding, companies, and investors.

    Args:
        query: Search query text
        top_k: Number of results to return (default: 10)
        date_filter: Temporal filter (e.g., "last_week", "this_month", "this_quarter")
        company_filter: List of company names to filter by
        investor_filter: List of investor names to filter by
        sector_filter: List of sectors/industries to filter by

    Returns:
        JSON string with search results including content, metadata, and citations
    """
    try:
        response = await _rag_client.query(
            query=query,
            top_k=top_k,
            date_filter=date_filter,
            company_filter=company_filter,
            investor_filter=investor_filter,
            sector_filter=sector_filter,
        )

        # Format results for agent consumption
        results = []
        for result in response.get("results", []):
            results.append({
                "content": result.get("content", ""),
                "metadata": result.get("metadata", {}),
                "document_id": result.get("document_id", ""),
                "chunk_id": result.get("chunk_id", ""),
                "similarity": result.get("similarity", 0.0),
                "rerank_score": result.get("rerank_score"),
            })

        return json.dumps({
            "results": results,
            "count": len(results),
            "search_mode": response.get("search_mode", "vector"),
        })
    except Exception as e:
        return json.dumps({"error": str(e), "results": [], "count": 0})

