"""Query rewriting service for generating query variations."""
import json
import logging
from typing import List, Optional
from app.services.llm_client import LLMServiceClient
from app.config import settings

logger = logging.getLogger(__name__)


class QueryRewriter:
    """Service for rewriting queries to improve retrieval."""

    def __init__(self):
        """Initialize query rewriter."""
        self.llm_client = LLMServiceClient()
        self.cache: dict[str, List[str]] = {}  # Simple in-memory cache

    async def rewrite_query(
        self,
        query: str,
        intent: str,
        user_id: Optional[str] = None,
    ) -> List[str]:
        """
        Generate query variations for better retrieval.

        Args:
            query: Original user query
            intent: Classified intent (research_query, trend_analysis, etc.)
            user_id: User ID for usage tracking

        Returns:
            List of query variations (includes original query)
        """
        # Check cache first
        cache_key = f"{query}:{intent}"
        if cache_key in self.cache:
            logger.debug(f"Using cached query variations for: {query}")
            return self.cache[cache_key]

        try:
            # Generate query variations using LLM
            messages = [
                {
                    "role": "system",
                    "content": """You are a query rewriting assistant for a startup/VC research system.

TASK: Generate 3-5 search query variations that would help find relevant articles about startups, funding, companies, and investors.

GUIDELINES:
- Keep the core intent of the original query
- Expand with synonyms and related terms (e.g., "AI" → "artificial intelligence", "startup" → "company" or "venture")
- Add domain-specific terms when relevant (e.g., "funding" → "funding round" or "venture capital")
- Include variations that might appear in article titles or content
- Make queries more specific when the original is vague

OUTPUT FORMAT:
Return a JSON object with a "queries" array containing 3-5 query strings.
Example: {"queries": ["query 1", "query 2", "query 3"]}""",
                },
                {
                    "role": "user",
                    "content": f"Original query: {query}\nIntent: {intent}\n\nGenerate 3-5 query variations that would find relevant startup/VC articles.",
                },
            ]

            # Use low temperature for deterministic query generation
            response = await self.llm_client.complete(
                messages=messages,
                temperature=0.2,
                user_id=user_id or "query-rewriter",
            )

            content = response.get("content", "").strip()

            # Parse JSON response
            try:
                # Try to extract JSON from response (might be wrapped in markdown code blocks)
                if "```json" in content:
                    json_start = content.find("```json") + 7
                    json_end = content.find("```", json_start)
                    content = content[json_start:json_end].strip()
                elif "```" in content:
                    json_start = content.find("```") + 3
                    json_end = content.find("```", json_start)
                    content = content[json_start:json_end].strip()

                parsed = json.loads(content)
                queries = parsed.get("queries", [])
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse query variations JSON: {e}. Content: {content}")
                # Fallback: try to extract queries from text
                queries = self._extract_queries_from_text(content)

            # Validate and clean queries
            if not queries or len(queries) == 0:
                logger.warning("No queries generated, using original query")
                queries = [query]
            else:
                # Ensure original query is included
                if query not in queries:
                    queries.insert(0, query)
                # Limit to 5 queries
                queries = queries[:5]

            # Cache results
            self.cache[cache_key] = queries
            logger.info(f"Generated {len(queries)} query variations for: {query}")

            return queries

        except Exception as e:
            logger.error(f"Query rewriting failed: {e}", exc_info=True)
            # Fallback to original query
            return [query]

    def _extract_queries_from_text(self, text: str) -> List[str]:
        """Extract queries from text if JSON parsing fails."""
        queries = []
        # Look for numbered lists or bullet points
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            # Remove numbering/bullets
            for prefix in ["-", "*", "1.", "2.", "3.", "4.", "5."]:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break
            # Remove quotes
            line = line.strip('"').strip("'").strip()
            if line and len(line) > 5:  # Minimum query length
                queries.append(line)
        return queries[:5] if queries else []

    def clear_cache(self):
        """Clear the query cache."""
        self.cache.clear()
        logger.info("Query rewriter cache cleared")

