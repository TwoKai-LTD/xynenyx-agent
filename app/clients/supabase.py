"""Supabase client wrapper for conversations and messages."""
from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime
from supabase import create_client, Client
from app.config import settings


class SupabaseClient:
    """Supabase client wrapper for agent operations."""

    def __init__(self):
        """Initialize Supabase client."""
        self.client: Client = create_client(
            settings.supabase_url,
            settings.supabase_key,
        )

    async def get_conversation(
        self,
        conversation_id: str,
        user_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get a conversation by ID, verifying ownership.

        Args:
            conversation_id: Conversation ID
            user_id: User ID for ownership verification

        Returns:
            Conversation dict or None if not found/unauthorized
        """
        try:
            result = (
                self.client.table("conversations")
                .select("*")
                .eq("id", conversation_id)
                .eq("user_id", user_id)
                .single()
                .execute()
            )
            return result.data if result.data else None
        except Exception:
            return None

    async def create_conversation(
        self,
        user_id: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new conversation.

        Args:
            user_id: User ID
            title: Conversation title (optional)
            metadata: Additional metadata

        Returns:
            Created conversation dict
        """
        result = (
            self.client.table("conversations")
            .insert({
                "user_id": user_id,
                "title": title or "New Conversation",
                "metadata": metadata or {},
            })
            .execute()
        )
        return result.data[0] if result.data else {}

    async def delete_conversation(
        self,
        conversation_id: str,
        user_id: str,
    ) -> bool:
        """
        Delete a conversation, verifying ownership.

        Args:
            conversation_id: Conversation ID
            user_id: User ID for ownership verification

        Returns:
            True if deleted, False otherwise
        """
        try:
            result = (
                self.client.table("conversations")
                .delete()
                .eq("id", conversation_id)
                .eq("user_id", user_id)
                .execute()
            )
            return len(result.data) > 0 if result.data else False
        except Exception:
            return False

    async def list_conversations(
        self,
        user_id: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        List conversations for a user.

        Args:
            user_id: User ID
            limit: Maximum number of conversations to return

        Returns:
            List of conversation dicts
        """
        result = (
            self.client.table("conversations")
            .select("*")
            .eq("user_id", user_id)
            .order("updated_at", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data if result.data else []

    async def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        tool_calls: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Save a message to the database.

        Args:
            conversation_id: Conversation ID
            role: Message role (user, assistant, system, tool)
            content: Message content
            sources: Source citations (optional)
            tool_calls: Tool calls (optional)
            metadata: Additional metadata (optional)

        Returns:
            Created message dict
        """
        result = (
            self.client.table("messages")
            .insert({
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
                "sources": sources or [],
                "tool_calls": tool_calls,
                "metadata": metadata or {},
            })
            .execute()
        )
        return result.data[0] if result.data else {}

    async def get_messages(
        self,
        conversation_id: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get messages for a conversation.

        Args:
            conversation_id: Conversation ID
            limit: Maximum number of messages to return

        Returns:
            List of message dicts
        """
        result = (
            self.client.table("messages")
            .select("*")
            .eq("conversation_id", conversation_id)
            .order("created_at", desc=False)
            .limit(limit)
            .execute()
        )
        return result.data if result.data else []

    async def update_conversation_title(
        self,
        conversation_id: str,
        user_id: str,
        title: str,
    ) -> bool:
        """
        Update conversation title.

        Args:
            conversation_id: Conversation ID
            user_id: User ID for ownership verification
            title: New title

        Returns:
            True if updated, False otherwise
        """
        try:
            result = (
                self.client.table("conversations")
                .update({"title": title, "updated_at": datetime.utcnow().isoformat()})
                .eq("id", conversation_id)
                .eq("user_id", user_id)
                .execute()
            )
            return len(result.data) > 0 if result.data else False
        except Exception:
            return False

