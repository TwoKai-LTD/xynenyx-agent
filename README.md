# Xynenyx Agent Service

LangGraph agent service that orchestrates multi-step AI workflows for startup and VC intelligence research.

## Overview

The agent service uses LangGraph to:

- Classify user intent (research_query, comparison, trend_analysis, temporal_query, entity_research, out_of_scope)
- Execute tools (RAG search, comparison, trend analysis, calculator)
- Manage conversation state with checkpoints
- Generate domain-specific responses with citations
- Stream responses to clients via SSE

## Architecture

The agent follows a graph-based workflow:

```
User Query → Classify Intent → [Retrieve Context | Execute Tools] → Generate Response → Stream/Save
```

### Graph Nodes

1. **classify_intent**: Uses LLM service to classify user intent
2. **retrieve_context**: Queries RAG service for relevant documents
3. **execute_tools**: Executes tools based on intent (RAG, comparison, trend analysis)
4. **generate_response**: Generates response using LLM with context
5. **handle_error**: Handles errors gracefully

### Tools

- **rag_search**: Search knowledge base with filters
- **compare_entities**: Compare companies/funding rounds
- **analyze_trends**: Analyze market trends and patterns
- **calculate**: Basic math operations and currency conversions

### State Persistence

- Checkpoints stored in Supabase `agent_checkpoints` table
- Conversation history in `conversations` and `messages` tables
- Thread ID = conversation_id for checkpoint management

## Quick Start

### Local Development

```bash
# Install dependencies
poetry install

# Set up environment variables (see .env.example)
cp .env.example .env

# Run locally
poetry run uvicorn app.main:app --port 8001 --reload
```

### Docker

```bash
docker build -t xynenyx-agent .
docker run -p 8001:8001 --env-file .env xynenyx-agent
```

## API Endpoints

### Health Checks

- `GET /health` - Health check
- `GET /ready` - Readiness check (verifies graph initialization)

### Chat

**POST /chat** - Synchronous chat completion

Request:
```json
{
  "message": "What is Xynenyx?",
  "conversation_id": "conv-123",
  "user_id": "user-456",
  "stream": false
}
```

Response:
```json
{
  "message": "Xynenyx is an AI-powered research assistant...",
  "conversation_id": "conv-123",
  "sources": [
    {
      "content": "...",
      "metadata": {...},
      "document_id": "doc-1",
      "chunk_id": "chunk-1"
    }
  ],
  "tools_used": ["rag_search"],
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "total_tokens": 150
  }
}
```

**POST /chat/stream** - Streaming chat (SSE)

Request: Same as `/chat` with `stream: true`

Response: Server-Sent Events stream
```
data: {"type": "token", "content": "X"}
data: {"type": "token", "content": "y"}
...
data: {"type": "end", "content": "", "sources": [...], "usage": {...}}
```

### Conversations

**GET /conversations** - List conversations

Headers:
- `X-User-ID: user-456`

Response:
```json
[
  {
    "id": "conv-123",
    "title": "New Conversation",
    "created_at": "2024-01-01T00:00:00",
    "updated_at": "2024-01-01T00:00:00",
    "metadata": {}
  }
]
```

**GET /conversations/{id}** - Get conversation

**POST /conversations** - Create conversation

Request:
```json
{
  "title": "My Conversation",
  "metadata": {}
}
```

**DELETE /conversations/{id}** - Delete conversation

## Configuration

Environment variables (see `.env.example`):

- `SUPABASE_URL` - Supabase project URL
- `SUPABASE_SERVICE_ROLE_KEY` - Supabase service role key
- `LLM_SERVICE_URL` - LLM service URL (default: http://localhost:8003)
- `RAG_SERVICE_URL` - RAG service URL (default: http://localhost:8002)
- `LLM_DEFAULT_PROVIDER` - Default LLM provider (default: openai)
- `LLM_DEFAULT_MODEL` - Default model (default: gpt-4o-mini)
- `CHECKPOINT_ENABLED` - Enable checkpointing (default: true)
- `CORS_ORIGINS` - CORS allowed origins (default: ["*"])

## Testing

```bash
# Run all tests
poetry run pytest -v

# Run with coverage
poetry run pytest --cov=app --cov-report=html

# Run specific test file
poetry run pytest tests/test_graph.py -v
```

## Implementation Details

### Intent Classification

The agent classifies user intent using the LLM service with a system prompt. Supported intents:

- `research_query`: General information requests
- `comparison`: Compare entities (companies, funding rounds)
- `trend_analysis`: Analyze market trends
- `temporal_query`: Time-based queries
- `entity_research`: Research specific companies/investors
- `out_of_scope`: Outside domain (redirect politely)

### Tool Execution

Tools are executed based on intent:
- `research_query`, `temporal_query`, `entity_research` → RAG tool
- `comparison` → Comparison tool
- `trend_analysis` → Trend tool

### State Management

- State persisted in Supabase checkpoints
- Conversation history loaded from database
- Messages saved after each interaction
- Checkpoints enable resumable conversations

### Error Handling

- Tool errors → `handle_error` node
- LLM service errors → Retry or fallback
- RAG service errors → Graceful degradation
- State persistence errors → Log and continue

## Dependencies

- Phase 2 (LLM Service) ✅
- Phase 4 (RAG Service) ✅
- Supabase database with tables:
  - `conversations`
  - `messages`
  - `agent_checkpoints`

## License

MIT License - see [LICENSE](LICENSE) file

