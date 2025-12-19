"""Comparison tool for comparing entities."""
from langchain_core.tools import tool
from typing import List, Dict, Any
import json
import re
from app.services.rag_client import RAGServiceClient

_rag_client = RAGServiceClient()


def extract_funding_data(content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Extract structured funding data from content and metadata."""
    data = {
        "funding_amount": None,
        "funding_round": None,
        "date": None,
        "investors": [],
        "valuation": None,
    }

    # Extract from metadata first
    if "funding_amount" in metadata:
        data["funding_amount"] = metadata["funding_amount"]
    if "funding_round" in metadata:
        data["funding_round"] = metadata["funding_round"]
    if "date" in metadata:
        data["date"] = metadata["date"]
    if "investors" in metadata:
        data["investors"] = metadata["investors"]

    # Extract from content if not in metadata
    if not data["funding_amount"]:
        # Look for funding amounts
        funding_patterns = [
            r"\$(\d+(?:\.\d+)?)\s*(?:million|M|billion|B)",
            r"raised\s+\$(\d+(?:\.\d+)?)\s*(?:million|M|billion|B)?",
        ]
        for pattern in funding_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                amount = float(match.group(1))
                if "billion" in match.group(0).lower() or "B" in match.group(0):
                    amount *= 1000
                data["funding_amount"] = amount
                break

    if not data["funding_round"]:
        round_patterns = [
            r"(?:Seed|seed)\s+round",
            r"Series\s+([A-Z])\s+round",
            r"Series\s+([A-Z])\s+funding",
        ]
        for pattern in round_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                if "seed" in match.group(0).lower():
                    data["funding_round"] = "Seed"
                else:
                    data["funding_round"] = f"Series {match.group(1)}"
                break

    if not data["investors"]:
        # Look for investor mentions
        investor_patterns = [
            r"led\s+by\s+([A-Z][a-zA-Z\s&]+)",
            r"investors\s+include\s+([A-Z][a-zA-Z\s,&]+)",
        ]
        for pattern in investor_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                investors = [inv.strip() for inv in match.group(1).split(",")]
                data["investors"] = investors[:5]  # Limit to 5
                break

    return data


@tool
async def compare_entities(
    entities: List[str],
    query_context: str = "",
) -> str:
    """
    Compare multiple entities (companies, funding rounds) by querying the knowledge base
    and extracting structured data for comparison.

    Args:
        entities: List of entity names to compare
        query_context: Additional context about what to compare

    Returns:
        JSON string with structured comparison data
    """
    try:
        comparison_data = {}

        # Query RAG for each entity
        for entity in entities:
            try:
                response = await _rag_client.query(
                    query=f"{entity} {query_context}",
                    top_k=5,
                    company_filter=[entity] if entity else None,
                )

                # Extract structured data from results
                entity_data = {
                    "name": entity,
                    "funding_rounds": [],
                    "total_funding": 0,
                    "latest_round": None,
                    "investors": set(),
                    "dates": [],
                }

                for result in response.get("results", []):
                    funding_data = extract_funding_data(
                        result.get("content", ""),
                        result.get("metadata", {}),
                    )

                    if funding_data["funding_amount"]:
                        entity_data["funding_rounds"].append(funding_data)
                        entity_data["total_funding"] += funding_data["funding_amount"] or 0
                        if funding_data["investors"]:
                            entity_data["investors"].update(funding_data["investors"])
                        if funding_data["date"]:
                            entity_data["dates"].append(funding_data["date"])

                # Get latest round
                if entity_data["funding_rounds"]:
                    entity_data["latest_round"] = max(
                        entity_data["funding_rounds"],
                        key=lambda x: x.get("date") or "",
                    )

                # Convert sets to lists for JSON serialization
                entity_data["investors"] = list(entity_data["investors"])[:10]
                comparison_data[entity] = entity_data

            except Exception as e:
                comparison_data[entity] = {"error": str(e)}

        return json.dumps(comparison_data)
    except Exception as e:
        return json.dumps({"error": str(e)})

