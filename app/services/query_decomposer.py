"""Query decomposition service for multi-part questions."""
import json
import logging
from typing import List, Dict, Any, Optional
from app.services.llm_client import LLMServiceClient

logger = logging.getLogger(__name__)


class QueryDecomposer:
    """Service for decomposing complex multi-part queries."""

    def __init__(self):
        """Initialize query decomposer."""
        self.llm_client = LLMServiceClient()

    def is_multi_part(self, query: str) -> bool:
        """
        Check if query is multi-part (contains multiple questions or comparisons).

        Args:
            query: User query

        Returns:
            True if query appears to be multi-part
        """
        # Simple heuristics for multi-part queries
        multi_part_indicators = [
            " and ",
            " or ",
            "compare",
            "versus",
            " vs ",
            ", and",
            ", then",
            "? and",
            "? then",
        ]
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in multi_part_indicators)

    async def decompose_query(
        self,
        query: str,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Decompose a multi-part query into sub-queries.

        Args:
            query: Original multi-part query
            user_id: User ID for usage tracking

        Returns:
            List of sub-query dicts with 'query' and 'type' fields
        """
        if not self.is_multi_part(query):
            return [{"query": query, "type": "single"}]

        try:
            messages = [
                {
                    "role": "system",
                    "content": """You are a query decomposition assistant for a startup/VC research system.

TASK: Break down a multi-part question into separate, independent sub-questions that can be answered individually.

GUIDELINES:
- Each sub-query should be a complete, standalone question
- Sub-queries should be independent (can be answered separately)
- Preserve the original intent of each part
- For comparisons, create separate queries for each entity being compared

OUTPUT FORMAT:
Return a JSON object with a "sub_queries" array:
{
  "sub_queries": [
    {"query": "First sub-question", "type": "research_query"},
    {"query": "Second sub-question", "type": "comparison"}
  ]
}

Query types: research_query, comparison, trend_analysis, entity_research, temporal_query""",
                },
                {
                    "role": "user",
                    "content": f"Decompose this query into sub-queries:\n{query}",
                },
            ]

            response = await self.llm_client.complete(
                messages=messages,
                temperature=0.3,  # Low temperature for deterministic decomposition
                response_format={"type": "json_object"},
                user_id=user_id or "query-decomposer",
            )

            content = response.get("content", "").strip()
            parsed = json.loads(content)
            sub_queries = parsed.get("sub_queries", [])

            if not sub_queries:
                logger.warning("No sub-queries generated, using original query")
                return [{"query": query, "type": "single"}]

            logger.info(f"Decomposed query into {len(sub_queries)} sub-queries")
            return sub_queries

        except Exception as e:
            logger.error(f"Query decomposition failed: {e}", exc_info=True)
            # Fallback to original query
            return [{"query": query, "type": "single"}]

    def merge_results(
        self,
        sub_query_results: List[Dict[str, Any]],
        original_query: str,
    ) -> Dict[str, Any]:
        """
        Merge results from multiple sub-queries.

        Args:
            sub_query_results: List of results from each sub-query
            original_query: Original multi-part query

        Returns:
            Merged results dict
        """
        # Combine all results
        all_results = []
        all_sources = []
        total_count = 0

        for result in sub_query_results:
            results = result.get("results", [])
            sources = result.get("sources", [])
            all_results.extend(results)
            all_sources.extend(sources)
            total_count += result.get("count", 0)

        # Deduplicate by chunk_id
        seen_chunk_ids = set()
        deduplicated_results = []
        deduplicated_sources = []

        for result in all_results:
            chunk_id = result.get("chunk_id") or result.get("id")
            if chunk_id and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                deduplicated_results.append(result)

        for source in all_sources:
            chunk_id = source.get("chunk_id") or source.get("id")
            if chunk_id and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                deduplicated_sources.append(source)

        return {
            "results": deduplicated_results,
            "sources": deduplicated_sources,
            "count": len(deduplicated_results),
            "sub_query_count": len(sub_query_results),
        }

