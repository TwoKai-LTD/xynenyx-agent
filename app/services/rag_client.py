"""HTTP client for RAG service."""
import httpx
import logging
from typing import List, Dict, Any, Optional
from uuid import UUID
from app.config import settings

logger = logging.getLogger(__name__)


class RAGServiceClient:
    """Client for interacting with the RAG service."""

    def __init__(self):
        """Initialize RAG service client."""
        self.base_url = settings.rag_service_url.rstrip("/")
        self.timeout = settings.rag_service_timeout
        self.default_top_k = settings.rag_default_top_k
        self.use_hybrid_search = settings.rag_use_hybrid_search
        self.use_reranking = settings.rag_use_reranking

    async def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        user_id: str = "agent-service",
        date_filter: Optional[str | Dict[str, str]] = None,
        company_filter: Optional[List[str]] = None,
        investor_filter: Optional[List[str]] = None,
        sector_filter: Optional[List[str]] = None,
        use_hybrid_search: Optional[bool] = None,
        use_reranking: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Query the RAG service.

        Args:
            query: Search query text
            top_k: Number of results to return (defaults to configured default)
            user_id: User ID for RAG service
            date_filter: Temporal filter (preset string or dict with start_date/end_date)
            company_filter: List of company names to filter by
            investor_filter: List of investor names to filter by
            sector_filter: List of sectors to filter by
            use_hybrid_search: Enable hybrid search (defaults to configured default)
            use_reranking: Enable reranking (defaults to configured default)

        Returns:
            Dict with 'query', 'results', 'count', 'search_mode', 'reranking_enabled'

        Raises:
            httpx.HTTPError: If request fails
        """
        url = f"{self.base_url}/query"
        headers = {"X-User-ID": user_id}

        payload = {
            "query": query,
            "top_k": top_k or self.default_top_k,
            "use_hybrid_search": use_hybrid_search if use_hybrid_search is not None else self.use_hybrid_search,
            "use_reranking": use_reranking if use_reranking is not None else self.use_reranking,
        }

        if date_filter:
            payload["date_filter"] = date_filter
        if company_filter:
            payload["company_filter"] = company_filter
        if investor_filter:
            payload["investor_filter"] = investor_filter
        if sector_filter:
            payload["sector_filter"] = sector_filter

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                logger.error(f"RAG service request failed: {e}")
                raise

