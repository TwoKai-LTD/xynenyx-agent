"""Trend analysis tool."""
from langchain_core.tools import tool
from typing import List, Dict, Any, Optional
import json
from collections import defaultdict
from app.services.rag_client import RAGServiceClient

_rag_client = RAGServiceClient()


@tool
async def analyze_trends(
    query: str,
    time_period: Optional[str] = None,
    sector_filter: Optional[List[str]] = None,
) -> str:
    """
    Analyze trends in the startup/VC space by querying the knowledge base
    and aggregating data by sector, geography, and time.

    Args:
        query: Query describing the trend to analyze
        time_period: Time period filter (e.g., "last_quarter", "this_year")
        sector_filter: List of sectors to focus on

    Returns:
        JSON string with trend analysis including patterns, metrics, and insights
    """
    try:
        # Query RAG with temporal filter
        response = await _rag_client.query(
            query=query,
            top_k=50,  # Get more results for trend analysis
            date_filter=time_period,
            sector_filter=sector_filter,
        )

        # Aggregate data
        sector_counts = defaultdict(int)
        sector_funding = defaultdict(float)
        geography_counts = defaultdict(int)
        round_types = defaultdict(int)
        total_funding = 0.0
        date_range = []

        for result in response.get("results", []):
            metadata = result.get("metadata", {})
            content = result.get("content", "")

            # Sector aggregation
            if "sector" in metadata:
                sector = metadata["sector"]
                sector_counts[sector] += 1
                if "funding_amount" in metadata:
                    funding = float(metadata.get("funding_amount", 0) or 0)
                    sector_funding[sector] += funding
                    total_funding += funding

            # Geography aggregation
            if "location" in metadata:
                geography_counts[metadata["location"]] += 1

            # Round type aggregation
            if "funding_round" in metadata:
                round_types[metadata["funding_round"]] += 1

            # Date tracking
            if "date" in metadata:
                date_range.append(metadata["date"])

        # Calculate percentages
        total_count = len(response.get("results", []))
        sector_percentages = {
            sector: (count / total_count * 100) if total_count > 0 else 0
            for sector, count in sector_counts.items()
        }

        # Identify top sectors
        top_sectors = sorted(
            sector_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        # Calculate growth metrics (simplified)
        trends = {
            "total_deals": total_count,
            "total_funding": total_funding,
            "average_funding": total_funding / total_count if total_count > 0 else 0,
            "top_sectors": [{"sector": s, "count": c, "percentage": sector_percentages.get(s, 0)} for s, c in top_sectors],
            "sector_funding": {sector: amount for sector, amount in sector_funding.items()},
            "geography_distribution": dict(geography_counts),
            "round_distribution": dict(round_types),
            "date_range": {
                "earliest": min(date_range) if date_range else None,
                "latest": max(date_range) if date_range else None,
            },
        }

        return json.dumps(trends)
    except Exception as e:
        return json.dumps({"error": str(e)})

