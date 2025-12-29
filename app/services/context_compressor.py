"""Context compression service for long contexts."""
import logging
from typing import List, Dict, Any
from app.services.llm_client import LLMServiceClient
from app.config import settings

logger = logging.getLogger(__name__)


class ContextCompressor:
    """Service for compressing long contexts to fit within token limits."""

    def __init__(self):
        """Initialize context compressor."""
        self.llm_client = LLMServiceClient()
        self.max_context_tokens = 4000  # Approximate token limit before compression

    async def compress_context(
        self,
        context: List[Dict[str, Any]],
        query: str,
        user_id: str = "agent-service",
    ) -> List[Dict[str, Any]]:
        """
        Compress context if it's too long.

        Args:
            context: List of context items (RAG results or tool results)
            query: Original user query
            user_id: User ID for usage tracking

        Returns:
            Compressed context list
        """
        if not context:
            return context

        # Estimate token count (rough approximation: 1 token ≈ 4 characters)
        total_chars = sum(
            len(str(item.get("content", ""))) for item in context
        )
        estimated_tokens = total_chars // 4

        if estimated_tokens <= self.max_context_tokens:
            logger.debug(f"Context within limits ({estimated_tokens} tokens), no compression needed")
            return context

        logger.info(f"Context too long ({estimated_tokens} tokens), compressing...")

        try:
            # Batch compress items to avoid sequential LLM calls
            # For performance, compress in parallel batches or use simpler truncation
            if len(context) > 5:
                # Too many items - use fast truncation instead of slow LLM compression
                logger.info(f"Too many items ({len(context)}), using fast truncation")
                compressed_items = []
                for item in context[:5]:  # Keep top 5 most relevant
                    content = str(item.get("content", ""))
                    compressed_items.append({
                        **item,
                        "content": content[:500] + "..." if len(content) > 500 else content,
                        "metadata": {**item.get("metadata", {}), "compressed": True, "compression_method": "truncation"},
                    })
                logger.info(f"Compressed context from {len(context)} to {len(compressed_items)} items (truncation)")
                return compressed_items
            
            # For smaller contexts, use LLM compression but batch if possible
            compressed_items = []
            for item in context:
                compressed = await self._compress_item(item, query, user_id)
                if compressed:
                    compressed_items.append(compressed)

            logger.info(f"Compressed context from {len(context)} to {len(compressed_items)} items")
            return compressed_items

        except Exception as e:
            logger.error(f"Context compression failed: {e}", exc_info=True)
            # Fallback: return top items by relevance with truncation
            return [
                {
                    **item,
                    "content": str(item.get("content", ""))[:500] + "...",
                    "metadata": {**item.get("metadata", {}), "compressed": True, "compression_method": "fallback_truncation"},
                }
                for item in context[:5]
            ]

    async def _compress_item(
        self,
        item: Dict[str, Any],
        query: str,
        user_id: str,
    ) -> Dict[str, Any] | None:
        """
        Compress a single context item by extracting key facts.

        Args:
            item: Context item to compress
            query: Original user query
            user_id: User ID for usage tracking

        Returns:
            Compressed context item or None if not relevant
        """
        content = item.get("content", "")
        metadata = item.get("metadata", {})

        if not content:
            return None

        # Extract key facts using LLM
        messages = [
            {
                "role": "system",
                "content": """You are a fact extraction assistant for startup/VC research.

TASK: Extract only the key facts from the provided content that are relevant to the user's query.

EXTRACT:
- Funding amounts and rounds
- Company names
- Dates
- Investors
- Sectors/industries
- Key milestones or announcements

OUTPUT FORMAT:
Return a JSON object with extracted facts:
{
  "summary": "Brief 1-2 sentence summary",
  "funding_amount": "$X million" or null,
  "company": "Company name" or null,
  "date": "YYYY-MM-DD" or null,
  "investors": ["Investor 1", "Investor 2"] or null,
  "sectors": ["Sector 1", "Sector 2"] or null,
  "key_points": ["Point 1", "Point 2"]
}""",
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nContent to extract facts from:\n{content[:1000]}",
            },
        ]

        try:
            response = await self.llm_client.complete(
                messages=messages,
                temperature=0.2,  # Low temperature for factual extraction
                response_format={"type": "json_object"},
                user_id=user_id,
            )

            import json
            facts = json.loads(response.get("content", "{}"))

            # Reconstruct compressed item
            compressed_content = facts.get("summary", content[:200])
            if facts.get("key_points"):
                compressed_content += "\n\nKey points: " + "; ".join(facts.get("key_points", [])[:3])

            # Preserve metadata and add extracted facts
            compressed_metadata = metadata.copy()
            compressed_metadata.update({
                "extracted_funding": facts.get("funding_amount"),
                "extracted_company": facts.get("company"),
                "extracted_date": facts.get("date"),
                "extracted_investors": facts.get("investors"),
                "extracted_sectors": facts.get("sectors"),
                "compressed": True,
            })

            return {
                "content": compressed_content,
                "metadata": compressed_metadata,
                "document_id": item.get("document_id"),
                "chunk_id": item.get("chunk_id"),
                "similarity": item.get("similarity", 0.0),
            }

        except Exception as e:
            logger.warning(f"Failed to compress item: {e}")
            # Fallback: truncate content
            return {
                **item,
                "content": content[:300] + "...",
                "metadata": {**metadata, "compressed": True, "compression_method": "truncation"},
            }

