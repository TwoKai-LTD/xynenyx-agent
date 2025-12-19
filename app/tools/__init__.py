"""Tool definitions for the agent."""
from langchain_core.tools import tool
from typing import List, Dict, Any
from app.tools.rag_tool import rag_search
from app.tools.comparison_tool import compare_entities
from app.tools.trend_tool import analyze_trends
from app.tools.calculator import calculate

# Export all tools for LangGraph
__all__ = ["rag_search", "compare_entities", "analyze_trends", "calculate"]

# List of all available tools
ALL_TOOLS = [rag_search, compare_entities, analyze_trends, calculate]

