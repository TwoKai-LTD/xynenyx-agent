"""Chat endpoints for agent service."""
import logging
import json
import uuid
from typing import AsyncGenerator
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.schemas.requests import ChatRequest
from app.schemas.responses import ChatResponse, StreamChunk
from app.graph.graph import get_agent_graph
from app.graph.state import AgentState
from app.clients.supabase import SupabaseClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])
supabase_client = SupabaseClient()


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Handle a synchronous chat request.

    Args:
        request: Chat request with message, conversation_id, user_id

    Returns:
        Chat response with message, sources, tools_used, usage
    """
    try:
        # Verify conversation exists and belongs to user
        conversation = await supabase_client.get_conversation(
            request.conversation_id,
            request.user_id,
        )
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Load conversation history
        messages_data = await supabase_client.get_messages(request.conversation_id)
        messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages_data
        ]

        # Add user message
        user_message = {"role": "user", "content": request.message}
        messages.append(user_message)

        # Save user message
        await supabase_client.save_message(
            conversation_id=request.conversation_id,
            role="user",
            content=request.message,
        )

        # Initialize state
        initial_state: AgentState = {
            "messages": messages,
            "user_id": request.user_id,
            "conversation_id": request.conversation_id,
            "intent": None,
            "context": [],
            "tools_used": [],
            "requires_human": False,
            "error": None,
            "sources": [],
            "usage": None,
        }

        # Run graph
        graph = get_agent_graph()
        config = {"configurable": {"thread_id": request.conversation_id}}
        final_state = await graph.ainvoke(initial_state, config=config)

        # Get assistant message
        assistant_messages = [
            msg for msg in final_state["messages"]
            if msg.get("role") == "assistant"
        ]
        if not assistant_messages:
            raise HTTPException(status_code=500, detail="No response generated")

        assistant_message = assistant_messages[-1]

        # Save assistant message
        await supabase_client.save_message(
            conversation_id=request.conversation_id,
            role="assistant",
            content=assistant_message.get("content", ""),
            sources=final_state.get("sources", []),
            metadata={"tools_used": final_state.get("tools_used", [])},
        )

        return ChatResponse(
            message=assistant_message.get("content", ""),
            conversation_id=request.conversation_id,
            sources=final_state.get("sources", []),
            tools_used=final_state.get("tools_used", []),
            usage=final_state.get("usage", {}),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """
    Handle a streaming chat request (SSE).

    Args:
        request: Chat request with message, conversation_id, user_id, stream=True

    Returns:
        StreamingResponse with SSE chunks
    """
    async def generate() -> AsyncGenerator[str, None]:
        try:
            # Verify conversation exists and belongs to user
            conversation = await supabase_client.get_conversation(
                request.conversation_id,
                request.user_id,
            )
            if not conversation:
                yield f"data: {json.dumps({'type': 'error', 'content': 'Conversation not found'})}\n\n"
                return

            # Load conversation history
            messages_data = await supabase_client.get_messages(request.conversation_id)
            messages = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in messages_data
            ]

            # Add user message
            user_message = {"role": "user", "content": request.message}
            messages.append(user_message)

            # Save user message
            await supabase_client.save_message(
                conversation_id=request.conversation_id,
                role="user",
                content=request.message,
            )

            # Initialize state
            initial_state: AgentState = {
                "messages": messages,
                "user_id": request.user_id,
                "conversation_id": request.conversation_id,
                "intent": None,
                "context": [],
                "tools_used": [],
                "requires_human": False,
                "error": None,
                "sources": [],
                "usage": None,
            }

            # Run graph with streaming
            graph = get_agent_graph()
            config = {"configurable": {"thread_id": request.conversation_id}}

            # Stream tokens from LLM during generation
            # Note: This is a simplified version - in production, you'd stream from generate_response node
            final_state = await graph.ainvoke(initial_state, config=config)

            # Get assistant message
            assistant_messages = [
                msg for msg in final_state["messages"]
                if msg.get("role") == "assistant"
            ]
            if not assistant_messages:
                yield f"data: {json.dumps({'type': 'error', 'content': 'No response generated'})}\n\n"
                return

            assistant_content = assistant_messages[-1].get("content", "")

            # Stream tokens (simplified - in production, stream during generation)
            for char in assistant_content:
                chunk = StreamChunk(type="token", content=char)
                yield f"data: {chunk.model_dump_json()}\n\n"

            # Send end chunk with sources and usage
            end_chunk = StreamChunk(
                type="end",
                content="",
                sources=final_state.get("sources", []),
                usage=final_state.get("usage", {}),
            )
            yield f"data: {end_chunk.model_dump_json()}\n\n"

            # Save assistant message
            await supabase_client.save_message(
                conversation_id=request.conversation_id,
                role="assistant",
                content=assistant_content,
                sources=final_state.get("sources", []),
                metadata={"tools_used": final_state.get("tools_used", [])},
            )

        except Exception as e:
            logger.error(f"Streaming chat request failed: {e}", exc_info=True)
            error_chunk = StreamChunk(type="error", content=str(e))
            yield f"data: {error_chunk.model_dump_json()}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )

