"""Structured query parameter extraction using LLM."""
import logging
import json
from typing import Dict, Any, Optional, List
from app.services.llm_client import LLMServiceClient
from app.config import settings

logger = logging.getLogger(__name__)

_extractor_client = LLMServiceClient()


class QueryExtractor:
    """Extract structured parameters from user queries."""

    async def extract_parameters(
        self,
        query: str,
        intent: str,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract structured parameters from a user query.

        Args:
            query: User query text
            intent: Classified intent
            user_id: User ID for usage tracking

        Returns:
            Dictionary with extracted parameters:
            - time_period: "last_30_days", "last_quarter", "this_year", "all_time", etc.
            - sector_filter: List of sectors
            - company_filter: List of companies
            - investor_filter: List of investors
            - date_range: {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"} or None
        """
        try:
            # Build extraction prompt based on intent
            if intent == "trend_analysis":
                system_prompt = """ROLE: You are a query parameter extraction assistant for trend analysis.

TASK: Extract structured parameters from the user's query.

EXTRACT:
1. Time Period: Determine if user wants recent/latest data or all-time data
   - If query mentions "latest", "recent", "current", "now", "this month", "this week" → "last_30_days"
   - If query mentions "last month", "past month" → "last_month"
   - If query mentions "last quarter", "past quarter" → "last_quarter"
   - If query mentions "this year", "2025" → "this_year"
   - If query mentions "last year", "2024" → "last_year"
   - If no time mentioned → "all_time"

2. Sectors: Extract any specific sectors mentioned (AI, FinTech, Healthcare, etc.)

3. Companies: Extract any specific company names mentioned

4. Investors: Extract any specific investor names mentioned

OUTPUT FORMAT:
Return a JSON object:
{
  "time_period": "last_30_days" | "last_month" | "last_quarter" | "this_year" | "last_year" | "all_time",
  "sector_filter": ["AI", "FinTech"] or null,
  "company_filter": ["Company Name"] or null,
  "investor_filter": ["Investor Name"] or null,
  "date_range": {"start": "2025-12-01", "end": "2025-12-31"} or null
}"""
            elif intent == "temporal_query":
                system_prompt = """ROLE: You are a query parameter extraction assistant for temporal queries.

TASK: Extract time period and date range from the user's query.

EXTRACT:
- Time period keywords: "last week", "last month", "last quarter", "this year", etc.
- Specific dates: "in December", "since 2025", "between X and Y"
- Date ranges: Start and end dates if specified

OUTPUT FORMAT:
Return a JSON object:
{
  "time_period": "last_week" | "last_month" | "last_quarter" | "this_year" | null,
  "date_range": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"} or null,
  "sector_filter": null,
  "company_filter": null,
  "investor_filter": null
}"""
            else:
                # For other intents, extract what's relevant
                system_prompt = """ROLE: You are a query parameter extraction assistant.

TASK: Extract structured parameters from the user's query.

EXTRACT:
- Companies mentioned
- Investors mentioned
- Sectors mentioned
- Time periods mentioned

OUTPUT FORMAT:
Return a JSON object:
{
  "time_period": null or time period string,
  "sector_filter": ["Sector"] or null,
  "company_filter": ["Company"] or null,
  "investor_filter": ["Investor"] or null,
  "date_range": null or {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
}"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ]

            response = await _extractor_client.complete(
                messages=messages,
                temperature=0.1,  # Low temperature for extraction
                response_format={"type": "json_object"},
                user_id=user_id,
            )

            content = response.get("content", "{}")
            try:
                extracted = json.loads(content)
                # Validate and normalize
                return {
                    "time_period": extracted.get("time_period"),
                    "sector_filter": extracted.get("sector_filter"),
                    "company_filter": extracted.get("company_filter"),
                    "investor_filter": extracted.get("investor_filter"),
                    "date_range": extracted.get("date_range"),
                }
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse extraction result: {content}")
                return self._default_parameters(intent)

        except Exception as e:
            logger.error(f"Parameter extraction failed: {e}")
            return self._default_parameters(intent)

    def _default_parameters(self, intent: str) -> Dict[str, Any]:
        """Return default parameters for an intent."""
        defaults = {
            "time_period": None,
            "sector_filter": None,
            "company_filter": None,
            "investor_filter": None,
            "date_range": None,
        }
        
        # For trend_analysis, default to all_time (not last_30_days)
        # Let the tool decide based on query content
        return defaults

