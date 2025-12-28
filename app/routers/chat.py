"""Chat endpoints for agent service."""
import logging
import json
import uuid
from typing import AsyncGenerator
from fastapi import APIRouter, HTTPException, Header
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
async def chat(
    request: ChatRequest,
    user_id: str = Header(..., alias="X-User-ID"),
) -> ChatResponse:
    """
    Handle a synchronous chat request.

    Args:
        request: Chat request with message, conversation_id, user_id

    Returns:
        Chat response with message, sources, tools_used, usage
    """
    try:
        # Create conversation if not provided
        conversation_id = request.conversation_id
        if not conversation_id:
            # Create new conversation
            conversation = await supabase_client.create_conversation(
                user_id=user_id,
                title=request.message[:50] if len(request.message) > 50 else request.message,
            )
            conversation_id = conversation["id"]
        else:
            # Verify conversation exists and belongs to user
            conversation = await supabase_client.get_conversation(
                conversation_id,
                user_id,
            )
            if not conversation:
                raise HTTPException(status_code=404, detail="Conversation not found")

        # Load conversation history
        messages_data = await supabase_client.get_messages(conversation_id, user_id=user_id)
        messages = []
        for msg in messages_data:
            # Convert database message format to LangChain message format
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                from langchain_core.messages import HumanMessage
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                from langchain_core.messages import AIMessage
                messages.append(AIMessage(content=content))
            elif role == "system":
                from langchain_core.messages import SystemMessage
                messages.append(SystemMessage(content=content))

        # Add user message as LangChain HumanMessage
        from langchain_core.messages import HumanMessage
        user_message = HumanMessage(content=request.message)
        messages.append(user_message)

        # Save user message
        await supabase_client.save_message(
            conversation_id=conversation_id,
            role="user",
            content=request.message,
        )

        # Initialize state
        initial_state: AgentState = {
            "messages": messages,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "intent": None,
            "context": [],
            "tools_used": [],
            "requires_human": False,
            "error": None,
            "sources": [],
            "usage": None,
            "reasoning": None,
            "validation": None,
            "validation_retried": False,
        }

        # Run graph
        graph = get_agent_graph()
        config = {"configurable": {"thread_id": conversation_id}}
        final_state = await graph.ainvoke(initial_state, config=config)

        # Get assistant message from LangChain messages
        from langchain_core.messages import AIMessage
        assistant_messages = [
            msg for msg in final_state["messages"]
            if isinstance(msg, AIMessage)
        ]
        if not assistant_messages:
            raise HTTPException(status_code=500, detail="No response generated")

        assistant_message = assistant_messages[-1]

        # Save assistant message
        assistant_content = assistant_message.content if hasattr(assistant_message, 'content') else str(assistant_message)
        await supabase_client.save_message(
            conversation_id=conversation_id,
            role="assistant",
            content=assistant_content,
            sources=final_state.get("sources", []),
            metadata={"tools_used": final_state.get("tools_used", [])},
        )

        # Ensure usage is a dict, not None
        usage = final_state.get("usage")
        if usage is None:
            usage = {}
        
        return ChatResponse(
            message=assistant_content,
            conversation_id=conversation_id,
            sources=final_state.get("sources", []),
            tools_used=final_state.get("tools_used", []),
            usage=usage,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    user_id: str = Header(..., alias="X-User-ID"),
) -> StreamingResponse:
    """
    Handle a streaming chat request (SSE).

    Args:
        request: Chat request with message, conversation_id, stream=True
        user_id: User ID from X-User-ID header

    Returns:
        StreamingResponse with SSE chunks
    """
    async def generate() -> AsyncGenerator[str, None]:
        try:
            # Create conversation if not provided
            conversation_id = request.conversation_id
            if not conversation_id:
                # Create new conversation
                conversation = await supabase_client.create_conversation(
                    user_id=user_id,
                    title=request.message[:50] if len(request.message) > 50 else request.message,
                )
                conversation_id = conversation["id"]
            else:
                # Verify conversation exists and belongs to user
                conversation = await supabase_client.get_conversation(
                    conversation_id,
                    user_id,
                )
                if not conversation:
                    yield f"data: {json.dumps({'type': 'error', 'content': 'Conversation not found'})}\n\n"
                    return

            # Load conversation history
            messages_data = await supabase_client.get_messages(conversation_id, user_id=user_id)
            messages = []
            for msg in messages_data:
                # Convert database message format to LangChain message format
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    from langchain_core.messages import HumanMessage
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    from langchain_core.messages import AIMessage
                    messages.append(AIMessage(content=content))
                elif role == "system":
                    from langchain_core.messages import SystemMessage
                    messages.append(SystemMessage(content=content))

            # Add user message as LangChain HumanMessage
            from langchain_core.messages import HumanMessage
            user_message = HumanMessage(content=request.message)
            messages.append(user_message)

            # Save user message
            await supabase_client.save_message(
                conversation_id=conversation_id,
                role="user",
                content=request.message,
            )

            # Initialize state
            initial_state: AgentState = {
                "messages": messages,
                "user_id": user_id,
                "conversation_id": conversation_id,
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
            config = {"configurable": {"thread_id": conversation_id}}

            # Stream tokens from LLM during generation
            # Note: This is a simplified version - in production, you'd stream from generate_response node
            final_state = await graph.ainvoke(initial_state, config=config)

            # Get assistant message from LangChain messages
            from langchain_core.messages import AIMessage
            assistant_messages = [
                msg for msg in final_state["messages"]
                if isinstance(msg, AIMessage)
            ]
            if not assistant_messages:
                yield f"data: {json.dumps({'type': 'error', 'content': 'No response generated'})}\n\n"
                return

            assistant_content = assistant_messages[-1].content if hasattr(assistant_messages[-1], 'content') else str(assistant_messages[-1])

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
                conversation_id=conversation_id,
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

