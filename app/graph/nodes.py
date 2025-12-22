"""LangGraph node functions."""
import logging
from typing import Dict, Any
from app.graph.state import AgentState
from app.services.llm_client import LLMServiceClient
from app.services.rag_client import RAGServiceClient
from app.tools import rag_search, compare_entities, analyze_trends

logger = logging.getLogger(__name__)

_llm_client = LLMServiceClient()
_rag_client = RAGServiceClient()


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
        user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
        if not user_messages:
            state["intent"] = "research_query"
            return state

        latest_message = user_messages[-1].content if hasattr(user_messages[-1], 'content') else str(user_messages[-1])
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
        user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
        if not user_messages:
            return state

        query = user_messages[-1].content if hasattr(user_messages[-1], 'content') else str(user_messages[-1])

        # Extract filters from intent if applicable
        date_filter = None
        company_filter = None
        investor_filter = None
        sector_filter = None

        if state.get("intent") == "temporal_query":
            # Try to extract date information from query
            # For now, use a simple heuristic
            if "last week" in query.lower() or "past week" in query.lower():
                date_filter = "last_week"
            elif "last month" in query.lower() or "past month" in query.lower():
                date_filter = "last_month"
            elif "last quarter" in query.lower() or "past quarter" in query.lower():
                date_filter = "last_quarter"
            elif "this year" in query.lower():
                date_filter = "this_year"

        # Query RAG service
        response = await _rag_client.query(
            query=query,
            user_id=state.get("user_id"),
            date_filter=date_filter,
            company_filter=company_filter,
            investor_filter=investor_filter,
            sector_filter=sector_filter,
        )

        # Store results in context
        results = response.get("results", [])
        state["context"] = results

        # Extract sources for citations
        sources = []
        for result in results:
            sources.append({
                "content": result.get("content", ""),
                "metadata": result.get("metadata", {}),
                "document_id": result.get("document_id", ""),
                "chunk_id": result.get("chunk_id", ""),
            })
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
        user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
        if not user_messages:
            return state

        query = user_messages[-1].content if hasattr(user_messages[-1], 'content') else str(user_messages[-1])
        tools_used = state.get("tools_used", [])

        if intent == "comparison":
            # Use comparison tool
            # Extract entity names from query (simplified)
            # In production, use NER or LLM to extract entities
            entities = []  # Placeholder - would extract from query
            result = await compare_entities.ainvoke({
                "entities": entities,
                "query_context": query,
            })
            state["context"] = [{"tool": "compare_entities", "result": result}]
            tools_used.append("compare_entities")

        elif intent == "trend_analysis":
            # Use trend tool
            result = await analyze_trends.ainvoke({
                "query": query,
                "time_period": None,  # Could extract from query
                "sector_filter": None,
            })
            state["context"] = [{"tool": "analyze_trends", "result": result}]
            tools_used.append("analyze_trends")

        elif intent in ["research_query", "temporal_query", "entity_research"]:
            # Use RAG tool
            result = await rag_search.ainvoke({
                "query": query,
                "top_k": 10,
            })
            state["context"] = [{"tool": "rag_search", "result": result}]
            tools_used.append("rag_search")

        state["tools_used"] = tools_used
        logger.info(f"Executed tools: {tools_used}")
        return state
    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        state["error"] = f"Tool execution failed: {str(e)}"
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

        # Add system prompt based on intent
        intent = state.get("intent", "research_query")
        if intent == "comparison":
            system_prompt = """You are a helpful assistant that compares companies, funding rounds, and trends.
            Use the provided comparison data to create a clear, structured comparison."""
        elif intent == "trend_analysis":
            system_prompt = """You are a helpful assistant that analyzes trends in the startup and VC space.
            Use the provided trend data to identify patterns and insights."""
        else:
            system_prompt = """You are a helpful assistant for startup and venture capital research.
            Answer questions using the provided context and cite sources when available.
            Be concise, accurate, and helpful."""

        messages.append({"role": "system", "content": system_prompt})

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
                content = msg.content if hasattr(msg, 'content') else str(msg)
                messages.append({"role": role, "content": content})
            else:
                # Already a dict
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                })

        # Add context to the last user message or as a separate system message
        context = state.get("context", [])
        if context:
            context_text = "Context from knowledge base:\n\n"
            if isinstance(context, list) and len(context) > 0:
                if isinstance(context[0], dict) and "tool" in context[0]:
                    # Tool result
                    context_text += context[0].get("result", "")
                else:
                    # RAG results
                    for i, item in enumerate(context[:5], 1):  # Limit to top 5
                        content = item.get("content", "") if isinstance(item, dict) else str(item)
                        metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
                        context_text += f"[{i}] {content}\n"
                        if metadata:
                            context_text += f"   Metadata: {metadata}\n\n"
            messages.append({"role": "system", "content": context_text})

        # Generate response
        response = await _llm_client.complete(
            messages=messages,
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

