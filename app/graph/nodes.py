"""LangGraph node functions."""

import logging
from typing import Dict, Any
from app.graph.state import AgentState
from app.services.llm_client import LLMServiceClient
from app.services.rag_client import RAGServiceClient
from app.services.query_rewriter import QueryRewriter
from app.services.context_compressor import ContextCompressor
from app.services.query_decomposer import QueryDecomposer
from app.services.query_extractor import QueryExtractor
from app.tools import rag_search, compare_entities, analyze_trends

logger = logging.getLogger(__name__)

_llm_client = LLMServiceClient()
_rag_client = RAGServiceClient()
_query_rewriter = QueryRewriter()
_context_compressor = ContextCompressor()
_query_decomposer = QueryDecomposer()
_query_extractor = QueryExtractor()


async def classify_intent(state: AgentState) -> AgentState:
    """
    Classify user intent from the latest message.

    Args:
        state: Current agent state

    Returns:
        Updated state with intent set
    """
    try:
        # Get the latest user message (LangChain HumanMessage)
        from langchain_core.messages import HumanMessage

        user_messages = [
            msg for msg in state["messages"] if isinstance(msg, HumanMessage)
        ]
        if not user_messages:
            state["intent"] = "research_query"
            return state

        latest_message = (
            user_messages[-1].content
            if hasattr(user_messages[-1], "content")
            else str(user_messages[-1])
        )
        intent = await _llm_client.classify_intent(
            message=latest_message,
            user_id=state.get("user_id"),
        )
        state["intent"] = intent
        logger.info(f"Classified intent: {intent}")
        return state
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        state["intent"] = "research_query"  # Fallback
        return state


async def retrieve_context(state: AgentState) -> AgentState:
    """
    Retrieve context from RAG service based on query.

    Args:
        state: Current agent state

    Returns:
        Updated state with context and sources
    """
    try:
        # Get the latest user message (LangChain HumanMessage)
        from langchain_core.messages import HumanMessage

        user_messages = [
            msg for msg in state["messages"] if isinstance(msg, HumanMessage)
        ]
        if not user_messages:
            return state

        query = (
            user_messages[-1].content
            if hasattr(user_messages[-1], "content")
            else str(user_messages[-1])
        )

        # Extract filters from query using structured extraction
        intent = state.get("intent", "research_query")
        params = await _query_extractor.extract_parameters(
            query=query,
            intent=intent,
            user_id=state.get("user_id"),
        )
        
        date_filter = params.get("time_period")
        company_filter = params.get("company_filter")
        investor_filter = params.get("investor_filter")
        sector_filter = params.get("sector_filter")

        # Check if query is multi-part and decompose if needed
        intent = state.get("intent", "research_query")
        if _query_decomposer.is_multi_part(query):
            logger.info("Detected multi-part query, decomposing...")
            sub_queries = await _query_decomposer.decompose_query(
                query=query,
                user_id=state.get("user_id"),
            )

            # Retrieve for each sub-query
            sub_query_results = []
            for sub_query_data in sub_queries:
                sub_query = sub_query_data.get("query", query)
                sub_intent = sub_query_data.get("type", intent)

                # Generate query variations for sub-query
                query_variations = await _query_rewriter.rewrite_query(
                    query=sub_query,
                    intent=sub_intent,
                    user_id=state.get("user_id"),
                )

                # Retrieve for sub-query
                sub_response = await _rag_client.query(
                    query=sub_query,
                    user_id=state.get("user_id"),
                    date_filter=date_filter,
                    company_filter=company_filter,
                    investor_filter=investor_filter,
                    sector_filter=sector_filter,
                    use_multi_query=len(query_variations) > 1,
                    query_variations=(
                        query_variations if len(query_variations) > 1 else None
                    ),
                )

                sub_query_results.append(
                    {
                        "results": sub_response.get("results", []),
                        "sources": [
                            {
                                "content": r.get("content", ""),
                                "metadata": r.get("metadata", {}),
                                "document_id": r.get("document_id", ""),
                                "chunk_id": r.get("chunk_id", ""),
                                # Explicitly include URL and date for easy citation
                                "article_url": (
                                    r.get("metadata", {}).get("article_url")
                                    or r.get("metadata", {}).get("url")
                                    or r.get("metadata", {}).get("source_url")
                                    or ""
                                ),
                                "published_date": (
                                    r.get("metadata", {}).get("published_date")
                                    or r.get("metadata", {}).get("date")
                                    or ""
                                ),
                            }
                            for r in sub_response.get("results", [])
                        ],
                        "count": len(sub_response.get("results", [])),
                    }
                )

            # Merge results from all sub-queries
            merged = _query_decomposer.merge_results(sub_query_results, query)
            results = merged["results"]
            sources = merged["sources"]

            logger.info(
                f"Decomposed query into {len(sub_queries)} sub-queries, merged {len(results)} results"
            )
        else:
            # Single query - use standard retrieval with query rewriting
            query_variations = await _query_rewriter.rewrite_query(
                query=query,
                intent=intent,
                user_id=state.get("user_id"),
            )
            logger.info(
                f"Generated {len(query_variations)} query variations: {query_variations}"
            )

            # Use multi-query retrieval if we have variations
            use_multi_query = len(query_variations) > 1

            # Query RAG service with multi-query support
            response = await _rag_client.query(
                query=query,
                user_id=state.get("user_id"),
                date_filter=date_filter,
                company_filter=company_filter,
                investor_filter=investor_filter,
                sector_filter=sector_filter,
                use_multi_query=use_multi_query,
                query_variations=query_variations if use_multi_query else None,
            )

            # Store results in context
            results = response.get("results", [])

            # Extract sources for citations
            sources = []
            for result in results:
                metadata = result.get("metadata", {})
                sources.append(
                    {
                        "content": result.get("content", ""),
                        "metadata": metadata,
                        "document_id": result.get("document_id", ""),
                        "chunk_id": result.get("chunk_id", ""),
                        # Explicitly include URL and date for easy citation
                        "article_url": (
                            metadata.get("article_url")
                            or metadata.get("url")
                            or metadata.get("source_url")
                            or ""
                        ),
                        "published_date": (
                            metadata.get("published_date") or metadata.get("date") or ""
                        ),
                    }
                )

        state["context"] = results
        state["sources"] = sources

        logger.info(f"Retrieved {len(results)} context documents")
        return state
    except Exception as e:
        logger.error(f"Context retrieval failed: {e}")
        state["error"] = f"Failed to retrieve context: {str(e)}"
        return state


async def execute_tools(state: AgentState) -> AgentState:
    """
    Execute tools based on classified intent.

    Args:
        state: Current agent state

    Returns:
        Updated state with tool results in context
    """
    try:
        intent = state.get("intent")
        from langchain_core.messages import HumanMessage

        user_messages = [
            msg for msg in state["messages"] if isinstance(msg, HumanMessage)
        ]
        if not user_messages:
            return state

        query = (
            user_messages[-1].content
            if hasattr(user_messages[-1], "content")
            else str(user_messages[-1])
        )
        tools_used = state.get("tools_used", [])

        if intent == "comparison":
            # Use comparison tool
            # Extract entity names from query (simplified)
            # In production, use NER or LLM to extract entities
            entities = []  # Placeholder - would extract from query
            result = await compare_entities.ainvoke(
                {
                    "entities": entities,
                    "query_context": query,
                }
            )
            state["context"] = [{"tool": "compare_entities", "result": result}]
            tools_used.append("compare_entities")

        elif intent == "trend_analysis":
            # Extract parameters from query using structured extraction
            params = await _query_extractor.extract_parameters(
                query=query,
                intent=intent,
                user_id=state.get("user_id"),
            )
            
            # Use trend tool with extracted parameters
            result = await analyze_trends.ainvoke(
                {
                    "query": query,
                    "time_period": params.get("time_period"),
                    "sector_filter": params.get("sector_filter"),
                }
            )
            state["context"] = [{"tool": "analyze_trends", "result": result}]
            tools_used.append("analyze_trends")

        elif intent in ["research_query", "temporal_query", "entity_research"]:
            # Use RAG tool
            result = await rag_search.ainvoke(
                {
                    "query": query,
                    "top_k": 10,
                }
            )
            state["context"] = [{"tool": "rag_search", "result": result}]
            tools_used.append("rag_search")

        state["tools_used"] = tools_used
        logger.info(f"Executed tools: {tools_used}")
        return state
    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        state["error"] = f"Tool execution failed: {str(e)}"
        return state


async def reasoning_step(state: AgentState) -> AgentState:
    """
    Perform chain-of-thought reasoning for complex queries.

    Args:
        state: Current agent state

    Returns:
        Updated state with reasoning added
    """
    try:
        intent = state.get("intent", "research_query")

        # Only use CoT for complex queries
        complex_intents = ["trend_analysis", "comparison"]
        if intent not in complex_intents:
            # Skip reasoning for simple queries
            return state

        # Get the latest user message
        from langchain_core.messages import HumanMessage

        user_messages = [
            msg for msg in state["messages"] if isinstance(msg, HumanMessage)
        ]
        if not user_messages:
            return state

        query = (
            user_messages[-1].content
            if hasattr(user_messages[-1], "content")
            else str(user_messages[-1])
        )

        # Format context for reasoning
        context = state.get("context", [])
        context_text = ""
        if context:
            if isinstance(context, list) and len(context) > 0:
                if isinstance(context[0], dict) and "tool" in context[0]:
                    context_text = context[0].get("result", "")
                else:
                    # RAG results
                    for i, item in enumerate(context[:5], 1):
                        content = (
                            item.get("content", "")
                            if isinstance(item, dict)
                            else str(item)
                        )
                        metadata = (
                            item.get("metadata", {}) if isinstance(item, dict) else {}
                        )
                        doc_name = metadata.get(
                            "document_name", metadata.get("title", "")
                        )
                        context_text += f"[{i}] {doc_name}: {content[:200]}...\n"

        # Build reasoning prompt
        messages = [
            {
                "role": "system",
                "content": """ROLE: You are a reasoning assistant that helps break down complex questions into step-by-step thinking.

TASK: Analyze the user's question and the provided context, then think through how to answer it step by step.

THINKING PROCESS:
1. What specific information is the user asking for?
2. What data points do I need to extract from the context?
3. How should I structure the answer?
4. What patterns or insights should I highlight?

OUTPUT FORMAT:
Provide your reasoning in this format:

Reasoning:
1. Question Analysis: [What the user is asking]
2. Required Data: [What information to extract from context]
3. Answer Structure: [How to organize the response]
4. Key Insights: [What patterns or insights to highlight]

Answer: [Your final answer based on the reasoning above]""",
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nContext:\n{context_text}\n\nThink step by step and provide your reasoning, then give the final answer.",
            },
        ]

        # Use moderate temperature for reasoning
        from app.config import settings

        reasoning_response = await _llm_client.complete(
            messages=messages,
            temperature=0.5,  # Moderate temperature for reasoning
            user_id=state.get("user_id"),
            conversation_id=state.get("conversation_id"),
        )

        reasoning_content = reasoning_response.get("content", "")

        # Store reasoning in state for use in generate_response
        state["reasoning"] = reasoning_content
        logger.info("Generated chain-of-thought reasoning")

        return state
    except Exception as e:
        logger.error(f"Reasoning step failed: {e}", exc_info=True)
        # Don't fail the whole pipeline if reasoning fails
        return state


async def generate_response(state: AgentState) -> AgentState:
    """
    Generate response using LLM service with context.

    Args:
        state: Current agent state

    Returns:
        Updated state with assistant message added
    """
    try:
        # Build messages for LLM
        messages = []

        # Add reasoning to context if available
        reasoning = state.get("reasoning")
        if reasoning:
            # Prepend reasoning to system prompt for better context
            reasoning_summary = (
                reasoning[:500] + "..." if len(reasoning) > 500 else reasoning
            )
            logger.info(f"Using reasoning in response generation: {reasoning_summary}")

        # Add system prompt based on intent
        intent = state.get("intent", "research_query")
        if intent == "comparison":
            system_prompt = """ROLE: You are Xynenyx, an AI research assistant that compares companies, funding rounds, and trends in the startup/VC space.

TASK: Compare entities by extracting structured data from the provided context and presenting it in a clear, organized format.

CONTEXT FORMAT:
Each source includes article title, URL, date, sectors, companies, funding amounts, investors, and other relevant metadata.

OUTPUT FORMAT:
- Extract structured data: funding amounts, rounds, dates, investors, valuations, milestones
- Present in tables or structured lists for easy comparison
- Always cite sources with [Source: URL, Date]
- Highlight key differences and similarities
- Include specific numbers and dates from context

IMPORTANT: The context below contains real data from recent articles. Use this data to create a clear, structured comparison."""
        elif intent == "trend_analysis":
            system_prompt = """ROLE: You are Xynenyx, an AI research assistant that analyzes trends and patterns in the startup and venture capital space.

TASK: Analyze trends by identifying patterns, themes, and quantitative insights from the provided context.

CONTEXT FORMAT:
- If context contains "TREND ANALYSIS DATA", this is aggregated database data (no individual article sources)
- If context contains "Source" sections, these are individual articles with URLs and dates

OUTPUT FORMAT:
- Identify patterns and themes across the data
- Provide quantitative insights (totals, averages, percentages, growth rates)
- Break down by sectors, funding rounds, geography, and time periods
- Use structured sections with clear headings
- Use BILLIONS for large amounts (>$1B), millions for smaller amounts
- Include specific numbers and calculations from context

CITATION REQUIREMENTS:
- For aggregated trend data (from database): Note that data is "aggregated from database" or "based on database analysis" - no specific article citations needed
- For individual article data: ALWAYS cite with [Source: URL, Date] format
- Example: "Total Deals: 536 (aggregated from database)" or "Company X raised $10M [Source: https://techcrunch.com/article, 2025-12-19]"
- When using trend analysis data, you can say "Based on analysis of 536 funding rounds in the database" instead of citing individual articles

IMPORTANT: 
- If context shows "TREND ANALYSIS DATA", the funding amounts are already in BILLIONS (use billions, not millions)
- The data is aggregated from the database, so cite it as "aggregated database analysis" rather than individual articles
- Use the exact numbers provided in the context
- Do not say there is no data if the context contains relevant information"""
        else:
            system_prompt = """ROLE: You are Xynenyx, an AI research assistant specialized in startup and venture capital intelligence.

TASK: Answer questions using provided context from recent startup/VC articles. Extract specific information including funding amounts, company names, dates, sectors, and investors.

CONTEXT FORMAT:
Each source includes:
- Article title
- URL (for citation)
- Publication date
- Sectors (if available)
- Companies mentioned (if available)
- Funding amounts (if available)

OUTPUT FORMAT:
- Start with a direct answer to the question
- Use bullet points for multiple items or comparisons
- Always cite sources with [Source: URL, Date] format
- Include specific numbers (funding amounts, dates, valuations)
- If context doesn't contain relevant information, clearly state this
- Be concise but thorough

IMPORTANT: The context provided below contains real information from recent startup/VC articles. You MUST use this context to answer the user's question. ALWAYS use the provided context to answer questions - do not say there is no data if context is provided."""

        # Add reasoning if available
        if reasoning:
            system_prompt += f"\n\nREASONING FROM PREVIOUS STEP:\n{reasoning}\n\nUse this reasoning to guide your answer, but ensure you cite sources from the context below."

        messages.append({"role": "system", "content": system_prompt})

        # Select adaptive temperature based on intent
        from app.config import settings

        intent_temperatures = settings.intent_temperature_mapping
        temperature = intent_temperatures.get(intent, settings.llm_default_temperature)
        logger.info(f"Using temperature {temperature} for intent: {intent}")

        # Add conversation history (convert LangChain messages to dict format for LLM client)
        for msg in state.get("messages", []):
            from langchain_core.messages import BaseMessage

            if isinstance(msg, BaseMessage):
                # Convert LangChain message to dict
                role_map = {
                    "HumanMessage": "user",
                    "AIMessage": "assistant",
                    "SystemMessage": "system",
                }
                msg_type = type(msg).__name__
                role = role_map.get(msg_type, "user")
                content = msg.content if hasattr(msg, "content") else str(msg)
                messages.append({"role": role, "content": content})
            else:
                # Already a dict
                messages.append(
                    {
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", ""),
                    }
                )

        # Add context to the last user message or as a separate system message
        context = state.get("context", [])

        # Compress context if too long
        if context:
            # Get user query for compression
            from langchain_core.messages import HumanMessage

            user_messages = [
                msg
                for msg in state.get("messages", [])
                if isinstance(msg, HumanMessage)
            ]
            user_query = (
                user_messages[-1].content
                if user_messages and hasattr(user_messages[-1], "content")
                else ""
            )

            context = await _context_compressor.compress_context(
                context=context,
                query=user_query,
                user_id=state.get("user_id", "agent-service"),
            )

        if context:
            context_text = "=== CONTEXT FROM KNOWLEDGE BASE ===\n\n"
            if isinstance(context, list) and len(context) > 0:
                if isinstance(context[0], dict) and "tool" in context[0]:
                    # Tool result - format it clearly with units
                    tool_result = context[0].get("result", "")
                    tool_name = context[0].get("tool", "")

                    # Parse JSON if it's a trend analysis result
                    if tool_name == "analyze_trends" and tool_result:
                        try:
                            import json

                            trends_data = json.loads(tool_result)
                            context_text += "=== TREND ANALYSIS DATA ===\n\n"

                            # Time period context
                            time_period = trends_data.get("time_period", "all_time")
                            if time_period != "all_time":
                                context_text += (
                                    f"Time Period: {time_period} (latest/recent data)\n"
                                )
                            else:
                                context_text += "Time Period: All available data\n"

                            context_text += (
                                f"Total Deals: {trends_data.get('total_deals', 0)}\n"
                            )
                            context_text += f"Total Funding: ${trends_data.get('total_funding_billions', 0)} billion\n"
                            context_text += f"Average Funding: ${trends_data.get('average_funding_billions', 0)} billion per deal\n\n"

                            # Growth metrics (if available)
                            if trends_data.get("growth_metrics"):
                                gm = trends_data["growth_metrics"]
                                context_text += "=== GROWTH TRENDS ===\n"
                                context_text += f"Previous Period: {gm.get('previous_period_deals', 0)} deals, ${gm.get('previous_period_funding_billions', 0)}B\n"
                                deals_growth = gm.get("deals_growth_percent", 0)
                                funding_growth = gm.get("funding_growth_percent", 0)
                                context_text += f"Deals Growth: {deals_growth:+.1f}% {'(increasing)' if deals_growth > 0 else '(decreasing)' if deals_growth < 0 else '(stable)'}\n"
                                context_text += f"Funding Growth: {funding_growth:+.1f}% {'(increasing)' if funding_growth > 0 else '(decreasing)' if funding_growth < 0 else '(stable)'}\n\n"

                            # Notable recent deals
                            if trends_data.get("notable_deals"):
                                context_text += "=== NOTABLE RECENT DEALS ===\n"
                                for deal in trends_data["notable_deals"][:5]:
                                    context_text += f"- ${deal.get('amount_billions', 0)}B on {deal.get('round_date', 'N/A')} ({deal.get('round_type', 'Unknown')} round)\n"
                                context_text += "\n"

                            if trends_data.get("top_sectors"):
                                context_text += "Top Sectors:\n"
                                for sector in trends_data["top_sectors"][:10]:
                                    context_text += f"- {sector.get('sector', 'Unknown')}: {sector.get('count', 0)} deals, ${sector.get('funding_billions', 0)}B ({sector.get('percentage', 0)}% of deals)\n"
                                context_text += "\n"

                            if trends_data.get("round_distribution"):
                                context_text += "Funding Round Distribution:\n"
                                for round_type, count in list(
                                    trends_data["round_distribution"].items()
                                )[:10]:
                                    context_text += (
                                        f"- {round_type or 'Unknown'}: {count} deals\n"
                                    )
                                context_text += "\n"

                            if trends_data.get("date_range"):
                                dr = trends_data["date_range"]
                                if dr.get("earliest") or dr.get("latest"):
                                    context_text += f"Date Range: {dr.get('earliest', 'N/A')} to {dr.get('latest', 'N/A')}\n\n"

                            context_text += "NOTE: This data is aggregated from the database. Use billions for amounts >$1B. Focus on RECENT trends and changes when time_period shows recent data.\n"
                        except Exception as e:
                            # Fallback to raw result if parsing fails
                            context_text += tool_result
                    else:
                        # Other tool results - use as-is
                        context_text += tool_result
                else:
                    # RAG results - format clearly
                    for i, item in enumerate(context[:5], 1):  # Limit to top 5
                        content = (
                            item.get("content", "")
                            if isinstance(item, dict)
                            else str(item)
                        )
                        metadata = (
                            item.get("metadata", {}) if isinstance(item, dict) else {}
                        )

                        # Extract key information from metadata
                        doc_name = metadata.get(
                            "document_name", metadata.get("title", "")
                        )
                        # Try multiple possible keys for URL
                        doc_url = (
                            metadata.get("article_url")
                            or metadata.get("url")
                            or metadata.get("source_url")
                            or ""
                        )
                        # Try multiple possible keys for date
                        published_date = (
                            metadata.get("published_date") or metadata.get("date") or ""
                        )
                        sectors = metadata.get("sectors", [])
                        companies = metadata.get("companies", [])
                        funding_amount = metadata.get(
                            "funding_amount", metadata.get("amount", "")
                        )

                        context_text += f"--- Source {i} ---\n"
                        if doc_name:
                            context_text += f"Article: {doc_name}\n"
                        if doc_url:
                            context_text += f"URL: {doc_url}\n"
                        if published_date:
                            context_text += f"Date: {published_date}\n"
                        if sectors:
                            context_text += f"Sectors: {', '.join(sectors) if isinstance(sectors, list) else sectors}\n"
                        if companies:
                            context_text += f"Companies: {', '.join(companies) if isinstance(companies, list) else companies}\n"
                        if funding_amount:
                            context_text += f"Funding: {funding_amount}\n"
                        context_text += f"Content: {content}\n\n"

            if len(context) > 5:
                context_text += f"\n[Note: Showing top 5 of {len(context)} results]\n"

            context_text += "\n=== END CONTEXT ===\n\n"

            # Check if this is tool data (no sources) or RAG data (has sources)
            if (
                isinstance(context, list)
                and len(context) > 0
                and isinstance(context[0], dict)
                and "tool" in context[0]
            ):
                context_text += "NOTE: The data above is aggregated from the database. For trend analysis data, cite as 'aggregated from database analysis' rather than individual articles. Use BILLIONS for amounts >$1B.\n"
            else:
                context_text += "CRITICAL CITATION REQUIREMENT: For every statistic, number, fact, or piece of information you mention in your response, you MUST include a citation in the format [Source: URL, Date]. Examples:\n"
                context_text += "- 'Total Deals: 543 [Source: https://techcrunch.com/article, 2025-12-19]'\n"
                context_text += "- 'AI sector raised $1.8B [Source: https://techcrunch.com/article, 2025-12-19]'\n"

            context_text += "Use the information above to answer the user's question. Extract specific details like funding amounts, company names, dates, and sectors from the context."

            messages.append({"role": "system", "content": context_text})

        # Generate response with adaptive temperature
        response = await _llm_client.complete(
            messages=messages,
            temperature=temperature,
            user_id=state.get("user_id"),
            conversation_id=state.get("conversation_id"),
        )

        # Add assistant message to state as LangChain AIMessage
        from langchain_core.messages import AIMessage

        assistant_message = AIMessage(content=response.get("content", ""))
        state["messages"].append(assistant_message)

        # Store usage information (ensure it's always a dict)
        usage = response.get("usage")
        if usage is None or not isinstance(usage, dict):
            usage = {}
        state["usage"] = usage

        # Store sources for citations
        if not state.get("sources"):
            state["sources"] = []

        logger.info("Generated response")
        return state
    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        state["error"] = f"Failed to generate response: {str(e)}"
        # Ensure usage is always a dict, not None
        if state.get("usage") is None:
            state["usage"] = {}
        return state


async def validate_response(state: AgentState) -> AgentState:
    """
    Validate the generated response for accuracy and citations.

    Args:
        state: Current agent state

    Returns:
        Updated state with validation results
    """
    try:
        # Get the assistant message
        from langchain_core.messages import AIMessage

        assistant_messages = [
            msg for msg in state["messages"] if isinstance(msg, AIMessage)
        ]
        if not assistant_messages:
            return state

        assistant_message = assistant_messages[-1]
        response_content = (
            assistant_message.content
            if hasattr(assistant_message, "content")
            else str(assistant_message)
        )

        # Get context for validation
        context = state.get("context", [])
        sources = state.get("sources", [])

        # Build validation prompt
        context_summary = ""
        if context:
            for i, item in enumerate(context[:3], 1):
                content = (
                    item.get("content", "") if isinstance(item, dict) else str(item)
                )
                context_summary += f"[{i}] {content[:150]}...\n"

        sources_summary = ""
        if sources:
            for i, source in enumerate(sources[:3], 1):
                doc_name = source.get("metadata", {}).get("document_name", "Unknown")
                sources_summary += f"[{i}] {doc_name}\n"

        messages = [
            {
                "role": "system",
                "content": """ROLE: You are a response validation assistant for a startup/VC research system.

TASK: Validate that the generated response correctly uses the provided context and sources.

CHECK FOR:
1. Citations: Does the response cite sources when using information from context?
2. Accuracy: Do numbers, dates, and facts match what's in the context?
3. Hallucinations: Is the response making up information not in the context?
4. Completeness: Does the response use available context or say "no data" when context exists?

OUTPUT FORMAT:
Return a JSON object:
{
  "is_valid": true/false,
  "issues": ["Issue 1", "Issue 2"] or [],
  "missing_citations": ["Fact 1 that needs citation", "Fact 2 that needs citation"] or [],
  "hallucinations": ["Made-up fact 1", "Made-up fact 2"] or [],
  "corrections_needed": true/false
}""",
            },
            {
                "role": "user",
                "content": f"""Generated Response:
{response_content}

Available Context:
{context_summary}

Available Sources:
{sources_summary}

Validate the response. Check if it correctly uses the context, cites sources, and doesn't hallucinate information.""",
            },
        ]

        # Use low temperature for validation
        validation_response = await _llm_client.complete(
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"},
            user_id=state.get("user_id"),
            conversation_id=state.get("conversation_id"),
        )

        import json

        validation_result = json.loads(validation_response.get("content", "{}"))

        is_valid = validation_result.get("is_valid", True)
        issues = validation_result.get("issues", [])
        corrections_needed = validation_result.get("corrections_needed", False)

        # Return only the fields we're updating, not the full state
        # This prevents LangGraph from trying to validate/replace the entire state
        # Note: validation_issues is not in the state schema, so we don't store it
        update = {
            "validation": validation_result,
        }

        # If corrections are needed, log the issues but don't regenerate to avoid state conflicts
        # TODO: Re-enable regeneration once we fix the message duplication issue
        if corrections_needed:
            logger.warning(
                f"Response validation found issues: {issues}. Validation retry disabled to prevent message duplication."
            )
            # Set validation_retried to True to prevent routing loop
            # Even though we're not actually retrying, we need to tell the router to go to END
            update["validation_retried"] = True
            # Don't store validation_issues - it's not in the state schema
            # The issues are already in validation_result["issues"]

        logger.info(f"Response validation: valid={is_valid}, issues={len(issues)}")
        return update

    except Exception as e:
        logger.error(f"Response validation failed: {e}", exc_info=True)
        # Don't fail the whole pipeline if validation fails
        return state


async def handle_error(state: AgentState) -> AgentState:
    """
    Handle errors and generate user-friendly error message.

    Args:
        state: Current agent state

    Returns:
        Updated state with error message
    """
    error = state.get("error", "An unknown error occurred")
    logger.error(f"Handling error: {error}")

    # Ensure usage is always a dict, not None
    if state.get("usage") is None:
        state["usage"] = {}

    # Generate user-friendly error message as LangChain AIMessage
    from langchain_core.messages import AIMessage

    error_message = AIMessage(
        content=f"I apologize, but I encountered an error while processing your request: {error}. Please try rephrasing your question or try again later."
    )
    state["messages"].append(error_message)

    # Clear error after handling
    state["error"] = None
    return state
