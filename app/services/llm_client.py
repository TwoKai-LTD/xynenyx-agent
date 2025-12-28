"""HTTP client for LLM service."""
import httpx
import json
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from app.config import settings

logger = logging.getLogger(__name__)


class LLMServiceClient:
    """Client for interacting with the LLM service."""

    def __init__(self):
        """Initialize LLM service client."""
        self.base_url = settings.llm_service_url.rstrip("/")
        self.timeout = settings.llm_service_timeout
        self.default_provider = settings.llm_default_provider
        self.default_model = settings.llm_default_model
        self.default_temperature = settings.llm_default_temperature

    async def complete(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        response_format: Optional[Dict[str, str]] = None,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a synchronous completion.

        Args:
            messages: List of message dicts with 'role' and 'content'
            provider: Provider ID (defaults to configured default)
            model: Model name (defaults to configured default)
            temperature: Sampling temperature (defaults to configured default)
            user_id: User ID for usage tracking
            conversation_id: Conversation ID for usage tracking

        Returns:
            Dict with 'content', 'provider', 'model', 'usage', 'metadata'

        Raises:
            httpx.HTTPError: If request fails
        """
        url = f"{self.base_url}/complete"
        headers = {}
        if user_id:
            headers["X-User-ID"] = user_id
        if conversation_id:
            headers["X-Conversation-ID"] = conversation_id

        payload = {
            "messages": messages,
            "provider": provider or self.default_provider,
            "model": model or self.default_model,
            "temperature": temperature if temperature is not None else self.default_temperature,
        }
        if response_format:
            payload["response_format"] = response_format

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                logger.error(f"LLM service request failed: {e}")
                raise

    async def complete_stream(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate a streaming completion (SSE).

        Args:
            messages: List of message dicts with 'role' and 'content'
            provider: Provider ID (defaults to configured default)
            model: Model name (defaults to configured default)
            temperature: Sampling temperature (defaults to configured default)
            user_id: User ID for usage tracking
            conversation_id: Conversation ID for usage tracking

        Yields:
            Dict chunks with 'type', 'content', 'usage', 'metadata'

        Raises:
            httpx.HTTPError: If request fails
        """
        url = f"{self.base_url}/complete/stream"
        headers = {"Accept": "text/event-stream"}
        if user_id:
            headers["X-User-ID"] = user_id
        if conversation_id:
            headers["X-Conversation-ID"] = conversation_id

        payload = {
            "messages": messages,
            "provider": provider or self.default_provider,
            "model": model or self.default_model,
            "temperature": temperature if temperature is not None else self.default_temperature,
        }

        async with httpx.AsyncClient(timeout=self.timeout * 5) as client:  # Longer timeout for streaming
            try:
                async with client.stream("POST", url, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]  # Remove "data: " prefix
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                yield chunk
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse SSE chunk: {data_str}")
                                continue
            except httpx.HTTPError as e:
                logger.error(f"LLM service streaming request failed: {e}")
                raise

    async def classify_intent(
        self,
        message: str,
        user_id: Optional[str] = None,
    ) -> str:
        """
        Classify user intent using LLM service.

        Args:
            message: User message to classify
            user_id: User ID for usage tracking

        Returns:
            Intent string (research_query, comparison, trend_analysis, etc.)

        Raises:
            httpx.HTTPError: If request fails
        """
        # Use intent_classification prompt
        messages = [
            {
                "role": "system",
                "content": """Classify the user's intent into one of:
- research_query: User wants information about startups, funding, companies, investors
- comparison: User wants to compare companies, funding rounds, or trends
- trend_analysis: User wants to understand market trends or patterns
- temporal_query: User asks about events in a specific time period
- entity_research: User wants information about a specific company or investor
- out_of_scope: User asks about topics outside startup/VC (redirect politely)

Respond with only the intent name.""",
            },
            {"role": "user", "content": message},
        ]

        try:
            response = await self.complete(
                messages=messages,
                temperature=0.1,  # Low temperature for classification
                user_id=user_id,
            )
            intent = response.get("content", "").strip().lower()
            # Validate intent
            valid_intents = [
                "research_query",
                "comparison",
                "trend_analysis",
                "temporal_query",
                "entity_research",
                "out_of_scope",
            ]
            if intent not in valid_intents:
                logger.warning(f"Invalid intent '{intent}', using fallback")
                return settings.intent_fallback
            return intent
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return settings.intent_fallback

