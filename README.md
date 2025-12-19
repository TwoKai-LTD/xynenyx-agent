# Xynenyx Agent Service

LangGraph agent service that orchestrates multi-step AI workflows for startup and VC intelligence research.

## Overview

The agent service uses LangGraph to:

- Classify user intent
- Execute tools (RAG search, comparison, trend analysis)
- Manage conversation state
- Generate domain-specific responses
- Stream responses to clients

## Quick Start

### Local Development

```bash
# Install dependencies
poetry install

# Run locally
poetry run uvicorn app.main:app --port 8001 --reload
```

### Docker

```bash
docker build -t xynenyx-agent .
docker run -p 8001:8001 --env-file .env xynenyx-agent
```

## API Endpoints

- `GET /health` - Health check
- `GET /ready` - Readiness check
- `POST /chat` - Synchronous chat
- `POST /chat/stream` - Streaming chat (SSE)
- `GET /conversations` - List conversations
- `POST /conversations` - Create conversation
- `DELETE /conversations/{id}` - Delete conversation

## Configuration

See `.env.example` for all configuration options.

## Testing

```bash
poetry run pytest -v
poetry run pytest --cov=app --cov-report=html
```

## License

MIT License - see [LICENSE](LICENSE) file

