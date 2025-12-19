"""Conversation management endpoints."""
import logging
from typing import List
from fastapi import APIRouter, HTTPException, Header
from app.schemas.requests import ConversationCreateRequest
from app.schemas.responses import ConversationResponse
from app.clients.supabase import SupabaseClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["conversations"])
supabase_client = SupabaseClient()


@router.get("", response_model=List[ConversationResponse])
async def list_conversations(
    user_id: str = Header(..., alias="X-User-ID"),
    limit: int = 100,
) -> List[ConversationResponse]:
    """
    List conversations for a user.

    Args:
        user_id: User ID (from header)
        limit: Maximum number of conversations to return

    Returns:
        List of conversation responses
    """
    try:
        conversations = await supabase_client.list_conversations(user_id, limit=limit)
        return [
            ConversationResponse(
                id=str(conv["id"]),
                title=conv.get("title", "New Conversation"),
                created_at=conv["created_at"],
                updated_at=conv["updated_at"],
                metadata=conv.get("metadata", {}),
            )
            for conv in conversations
        ]
    except Exception as e:
        logger.error(f"Failed to list conversations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    user_id: str = Header(..., alias="X-User-ID"),
) -> ConversationResponse:
    """
    Get a conversation by ID.

    Args:
        conversation_id: Conversation ID
        user_id: User ID (from header)

    Returns:
        Conversation response with messages
    """
    try:
        conversation = await supabase_client.get_conversation(conversation_id, user_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Get messages
        messages = await supabase_client.get_messages(conversation_id)

        return ConversationResponse(
            id=str(conversation["id"]),
            title=conversation.get("title", "New Conversation"),
            created_at=conversation["created_at"],
            updated_at=conversation["updated_at"],
            metadata={
                **conversation.get("metadata", {}),
                "message_count": len(messages),
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("", response_model=ConversationResponse)
async def create_conversation(
    request: ConversationCreateRequest,
    user_id: str = Header(..., alias="X-User-ID"),
) -> ConversationResponse:
    """
    Create a new conversation.

    Args:
        request: Conversation creation request
        user_id: User ID (from header)

    Returns:
        Created conversation response
    """
    try:
        conversation = await supabase_client.create_conversation(
            user_id=user_id,
            title=request.title,
            metadata=request.metadata,
        )
        return ConversationResponse(
            id=str(conversation["id"]),
            title=conversation.get("title", "New Conversation"),
            created_at=conversation["created_at"],
            updated_at=conversation["updated_at"],
            metadata=conversation.get("metadata", {}),
        )
    except Exception as e:
        logger.error(f"Failed to create conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    user_id: str = Header(..., alias="X-User-ID"),
) -> dict:
    """
    Delete a conversation.

    Args:
        conversation_id: Conversation ID
        user_id: User ID (from header)

    Returns:
        Success message
    """
    try:
        deleted = await supabase_client.delete_conversation(conversation_id, user_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"message": "Conversation deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

