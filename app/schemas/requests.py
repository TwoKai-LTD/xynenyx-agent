"""Pydantic schemas for API requests."""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class ChatRequest(BaseModel):
    """Request for a chat completion."""

    message: str = Field(..., description="User message")
    conversation_id: str = Field(..., description="Conversation identifier")
    user_id: str = Field(..., description="User identifier")
    stream: bool = Field(default=False, description="Enable streaming response")


class ConversationCreateRequest(BaseModel):
    """Request to create a new conversation."""

    title: Optional[str] = Field(None, description="Conversation title")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

