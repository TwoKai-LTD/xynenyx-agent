"""Pydantic schemas for API responses."""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class ChatResponse(BaseModel):
    """Response from a chat completion."""

    message: str = Field(..., description="Assistant response message")
    conversation_id: str = Field(..., description="Conversation identifier")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Source citations")
    tools_used: List[str] = Field(default_factory=list, description="Tools used in processing")
    usage: Dict[str, int] = Field(default_factory=dict, description="Token usage information")


class StreamChunk(BaseModel):
    """A chunk from a streaming chat response."""

    type: str = Field(..., description="Chunk type: 'token', 'end', or 'error'")
    content: str = Field(default="", description="Chunk content (for token type)")
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="Source citations (for end type)")
    usage: Optional[Dict[str, int]] = Field(None, description="Token usage (for end type)")


class ConversationResponse(BaseModel):
    """Response for conversation information."""

    id: str = Field(..., description="Conversation ID")
    title: str = Field(..., description="Conversation title")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

