"""Trend analysis tool."""
from langchain_core.tools import tool
from typing import List, Dict, Any, Optional
import json
from datetime import datetime, timedelta
from app.clients.supabase import SupabaseClient

_supabase_client = SupabaseClient()


def _parse_time_period(time_period: Optional[str]) -> Optional[datetime]:
    """Parse time period string into a datetime for filtering."""
    if not time_period:
        return None
    
    now = datetime.utcnow()
    time_period_lower = time_period.lower()
    
    if "last_week" in time_period_lower or "past_week" in time_period_lower:
        return now - timedelta(days=7)
    elif "last_month" in time_period_lower or "past_month" in time_period_lower:
        return now - timedelta(days=30)
    elif "last_quarter" in time_period_lower or "past_quarter" in time_period_lower:
        return now - timedelta(days=90)
    elif "this_year" in time_period_lower:
        return datetime(now.year, 1, 1)
    elif "last_year" in time_period_lower or "past_year" in time_period_lower:
        return now - timedelta(days=365)
    
    return None


@tool
async def analyze_trends(
    query: str,
    time_period: Optional[str] = None,
    sector_filter: Optional[List[str]] = None,
) -> str:
    """
    Analyze trends in the startup/VC space by querying the database directly
    and aggregating data by sector, geography, and time.

    Args:
        query: Query describing the trend to analyze
        time_period: Time period filter (e.g., "last_quarter", "this_year")
        sector_filter: List of sectors to focus on

    Returns:
        JSON string with trend analysis including patterns, metrics, and insights
    """
    try:
        # Parse time period (now comes from structured extraction, not keyword matching)
        date_filter = _parse_time_period(time_period)
        date_filter_str = date_filter.date().isoformat() if date_filter else None

        # Query funding rounds with filters
        db_query = _supabase_client.client.table("funding_rounds").select("id, amount_usd, round_type, round_date, company_id, document_id")
        
        # Apply filters
        if date_filter_str:
            db_query = db_query.gte("round_date", date_filter_str)
        
        # Only get rounds with funding
        db_query = db_query.gt("amount_usd", 0)
        
        # Execute query
        result = db_query.execute()
        funding_rounds = result.data if result.data else []
        
        # Get comparison data (previous period) for trend analysis
        comparison_data = {}
        if date_filter_str and date_filter:
            # Get data from previous period for comparison (same length as current period)
            period_days = (datetime.utcnow().date() - date_filter.date()).days if date_filter else 30
            prev_period_start = date_filter - timedelta(days=period_days)
            if prev_period_start:
                prev_query = _supabase_client.client.table("funding_rounds").select("id, amount_usd, round_date")
                prev_query = prev_query.gte("round_date", prev_period_start.date().isoformat())
                prev_query = prev_query.lt("round_date", date_filter_str)
                prev_query = prev_query.gt("amount_usd", 0)
                prev_result = prev_query.execute()
                prev_rounds = prev_result.data if prev_result.data else []
                comparison_data = {
                    "previous_period_deals": len(prev_rounds),
                    "previous_period_funding": sum(float(r.get("amount_usd", 0) or 0) for r in prev_rounds),
                    "previous_period_days": period_days,
                }

        # Get document IDs to fetch sectors
        document_ids = list(set(r.get("document_id") for r in funding_rounds if r.get("document_id")))
        
        # Fetch document features for sectors (batch query)
        sectors_map = {}
        if document_ids:
            # Query in batches of 100 (Supabase PostgREST limit)
            batch_size = 100
            for i in range(0, len(document_ids), batch_size):
                batch = document_ids[i:i + batch_size]
                try:
                    # Use 'in' filter for batch query
                    df_result = _supabase_client.client.table("document_features").select("document_id, sectors").in_("document_id", batch).execute()
                    if df_result.data:
                        for df in df_result.data:
                            sectors_map[df["document_id"]] = df.get("sectors", [])
                except Exception:
                    # If batch query fails, fall back to individual queries
                    for doc_id in batch:
                        try:
                            df_result = _supabase_client.client.table("document_features").select("document_id, sectors").eq("document_id", doc_id).execute()
                            if df_result.data:
                                for df in df_result.data:
                                    sectors_map[df["document_id"]] = df.get("sectors", [])
                        except Exception:
                            continue

        # Filter by sectors if provided
        if sector_filter:
            filtered_rounds = []
            for round_data in funding_rounds:
                doc_id = round_data.get("document_id")
                sectors = sectors_map.get(doc_id, [])
                if sectors and any(s in sectors for s in sector_filter):
                    filtered_rounds.append(round_data)
            funding_rounds = filtered_rounds

        # Get company IDs for geography
        company_ids = list(set(r.get("company_id") for r in funding_rounds if r.get("company_id")))
        
        # Fetch companies for geography data (batch query)
        companies_map = {}
        if company_ids:
            # Query in batches of 100
            batch_size = 100
            for i in range(0, len(company_ids), batch_size):
                batch = company_ids[i:i + batch_size]
                try:
                    # Use 'in' filter for batch query
                    comp_result = _supabase_client.client.table("companies").select("id, metadata").in_("id", batch).execute()
                    if comp_result.data:
                        for comp in comp_result.data:
                            companies_map[comp["id"]] = comp.get("metadata", {})
                except Exception:
                    # If batch query fails, fall back to individual queries
                    for company_id in batch:
                        try:
                            comp_result = _supabase_client.client.table("companies").select("id, metadata").eq("id", company_id).execute()
                            if comp_result.data:
                                for comp in comp_result.data:
                                    companies_map[comp["id"]] = comp.get("metadata", {})
                        except Exception:
                            continue

        # Aggregate data
        total_deals = len(funding_rounds)
        total_funding = sum(float(r.get("amount_usd", 0) or 0) for r in funding_rounds)
        average_funding = total_funding / total_deals if total_deals > 0 else 0.0
        
        # Sector aggregation
        sector_counts = {}
        sector_funding = {}
        for round_data in funding_rounds:
            doc_id = round_data.get("document_id")
            sectors = sectors_map.get(doc_id, [])
            
            amount = float(round_data.get("amount_usd", 0) or 0)
            
            if sectors:
                for sector in sectors:
                    if sector:
                        sector_counts[sector] = sector_counts.get(sector, 0) + 1
                        sector_funding[sector] = sector_funding.get(sector, 0.0) + amount
            else:
                # If no sectors, use "Unknown"
                sector_counts["Unknown"] = sector_counts.get("Unknown", 0) + 1
                sector_funding["Unknown"] = sector_funding.get("Unknown", 0.0) + amount

        # Round type aggregation
        round_distribution = {}
        for round_data in funding_rounds:
            round_type = round_data.get("round_type") or "Unknown"
            round_distribution[round_type] = round_distribution.get(round_type, 0) + 1

        # Geography aggregation (from companies metadata)
        geography_distribution = {}
        for round_data in funding_rounds:
            company_id = round_data.get("company_id")
            if company_id:
                metadata = companies_map.get(company_id, {})
                location = metadata.get("location") if isinstance(metadata, dict) else None
                if location:
                    geography_distribution[location] = geography_distribution.get(location, 0) + 1

        # Date range
        dates = [r.get("round_date") for r in funding_rounds if r.get("round_date")]
        date_range = {
            "earliest": min(dates) if dates else None,
            "latest": max(dates) if dates else None,
        }

        # Top sectors (by count)
        top_sectors_list = sorted(
            sector_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        
        top_sectors = []
        for sector, count in top_sectors_list:
            percentage = (count / total_deals * 100) if total_deals > 0 else 0
            top_sectors.append({
                "sector": sector,
                "count": count,
                "percentage": round(percentage, 2),
                "funding_millions": round(sector_funding.get(sector, 0.0) / 1_000_000, 2),
            })

        # Calculate growth metrics if comparison data available
        growth_metrics = {}
        if comparison_data:
            prev_deals = comparison_data.get("previous_period_deals", 0)
            prev_funding = comparison_data.get("previous_period_funding", 0)
            if prev_deals > 0:
                growth_metrics = {
                    "deals_growth_percent": round(((total_deals - prev_deals) / prev_deals * 100), 1) if prev_deals > 0 else 0,
                    "funding_growth_percent": round(((total_funding - prev_funding) / prev_funding * 100), 1) if prev_funding > 0 else 0,
                    "previous_period_deals": prev_deals,
                    "previous_period_funding_billions": round(prev_funding / 1_000_000_000, 2),
                }
        
        # Get recent notable deals (top 5 by amount)
        notable_deals = []
        if funding_rounds:
            sorted_rounds = sorted(funding_rounds, key=lambda x: float(x.get("amount_usd", 0) or 0), reverse=True)
            for round_data in sorted_rounds[:5]:
                notable_deals.append({
                    "amount_billions": round(float(round_data.get("amount_usd", 0) or 0) / 1_000_000_000, 2),
                    "round_date": round_data.get("round_date"),
                    "round_type": round_data.get("round_type"),
                })
        
        # Convert funding to billions for readability and clarity
        # Store in billions to avoid confusion (LLM will interpret correctly)
        trends = {
            "total_deals": total_deals,
            "total_funding_billions": round(total_funding / 1_000_000_000, 2),  # Convert to billions
            "total_funding_millions": round(total_funding / 1_000_000, 2),  # Also provide in millions for reference
            "average_funding_billions": round(average_funding / 1_000_000_000, 2),  # Convert to billions
            "average_funding_millions": round(average_funding / 1_000_000, 2),  # Also provide in millions
            "top_sectors": [
                {
                    "sector": s["sector"],
                    "count": s["count"],
                    "percentage": s["percentage"],
                    "funding_billions": round(s["funding_millions"] / 1000, 2),  # Convert to billions
                    "funding_millions": s["funding_millions"],
                }
                for s in top_sectors
            ],
            "sector_funding_billions": {sector: round(amount / 1_000_000_000, 2) for sector, amount in sector_funding.items()},
            "sector_funding_millions": {sector: round(amount / 1_000_000, 2) for sector, amount in sector_funding.items()},
            "geography_distribution": geography_distribution,
            "round_distribution": round_distribution,
            "date_range": date_range,
            "time_period": date_filter_str or "all_time",
            "growth_metrics": growth_metrics if growth_metrics else None,
            "notable_deals": notable_deals,
            "note": "All funding amounts are in USD. Use billions for large amounts (>$1B) and millions for smaller amounts.",
        }

        return json.dumps(trends)
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Trend analysis failed: {e}", exc_info=True)
        return json.dumps({"error": str(e), "total_deals": 0, "total_funding": 0.0, "average_funding": 0.0, "top_sectors": [], "sector_funding": {}, "geography_distribution": {}, "round_distribution": {}, "date_range": {"earliest": None, "latest": None}})

