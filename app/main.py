"""FastAPI application for Xynenyx Agent Service."""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from app.config import settings
from app.routers import chat, conversations
from app.graph.graph import get_agent_graph
from app.schemas.errors import create_error_response
from app.middleware.logging import LoggingMiddleware

# Configure structured logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    logger.info("Starting Xynenyx Agent Service...")
    # Initialize graph
    try:
        graph = get_agent_graph()
        logger.info("Agent graph initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agent graph: {e}", exc_info=True)
        # Don't raise - allow service to start even if graph init fails
        # The /ready endpoint will report this issue
        logger.warning("Service will start but graph initialization failed - check /ready endpoint")

    yield

    # Shutdown
    logger.info("Shutting down Xynenyx Agent Service...")


app = FastAPI(
    title="Xynenyx Agent Service",
    version="0.1.0",
    lifespan=lifespan,
)

# Add logging middleware (before CORS to capture all requests)
app.add_middleware(LoggingMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# Include routers
app.include_router(chat.router)
app.include_router(conversations.router)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/ready")
async def ready():
    """Readiness check endpoint with dependency verification."""
    checks = {}
    all_ready = True

    # Check graph initialization
    try:
        graph = get_agent_graph()
        checks["graph"] = "ready"
    except Exception as e:
        logger.error(f"Graph initialization check failed: {e}")
        checks["graph"] = f"error: {str(e)}"
        all_ready = False

    # Check Supabase connection
    try:
        from app.clients.supabase import SupabaseClient
        client = SupabaseClient()
        # Simple query to verify connection
        result = client.client.table("conversations").select("id").limit(1).execute()
        checks["supabase"] = "ready"
    except Exception as e:
        logger.error(f"Supabase connection check failed: {e}")
        checks["supabase"] = f"error: {str(e)}"
        all_ready = False

    # Check LLM service
    try:
        import httpx
        async with httpx.AsyncClient(timeout=2.0) as http_client:
            response = await http_client.get(f"{settings.llm_service_url}/health")
            if response.status_code == 200:
                checks["llm_service"] = "ready"
            else:
                checks["llm_service"] = f"unhealthy: {response.status_code}"
                all_ready = False
    except Exception as e:
        logger.error(f"LLM service check failed: {e}")
        checks["llm_service"] = f"error: {str(e)}"
        all_ready = False

    # Check RAG service
    try:
        import httpx
        async with httpx.AsyncClient(timeout=2.0) as http_client:
            response = await http_client.get(f"{settings.rag_service_url}/health")
            if response.status_code == 200:
                checks["rag_service"] = "ready"
            else:
                checks["rag_service"] = f"unhealthy: {response.status_code}"
                all_ready = False
    except Exception as e:
        logger.error(f"RAG service check failed: {e}")
        checks["rag_service"] = f"error: {str(e)}"
        all_ready = False

    status_code = 200 if all_ready else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if all_ready else "not ready",
            "checks": checks,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    errors = exc.errors()
    return JSONResponse(
        status_code=422,
        content=create_error_response(
            detail="Validation error",
            status_code=422,
            code="VALIDATION_ERROR",
            errors=errors,
        ),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=create_error_response(
            detail=exc.detail,
            status_code=exc.status_code,
            code=f"HTTP_{exc.status_code}",
        ),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=create_error_response(
            detail="Internal server error",
            status_code=500,
            code="INTERNAL_SERVER_ERROR",
        ),
    )

