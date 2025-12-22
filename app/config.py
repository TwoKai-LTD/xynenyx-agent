"""Configuration settings for Xynenyx Agent Service."""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator, field_validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Service settings
    app_name: str = "Xynenyx Agent Service"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8001

    # Supabase settings
    supabase_url: str
    supabase_service_role_key: str | None = None
    # Support alternative name from .env (SUPABASE_SERVICE_KEY)
    supabase_service_key: str | None = None

    # LLM Service settings
    llm_service_url: str = "http://localhost:8003"
    llm_service_timeout: int = 60
    llm_default_provider: str = "openai"
    llm_default_model: str = "gpt-4o-mini"
    llm_default_temperature: float = 0.7

    # RAG Service settings
    rag_service_url: str = "http://localhost:8002"
    rag_service_timeout: int = 60
    rag_default_top_k: int = 10
    rag_use_hybrid_search: bool = True
    rag_use_reranking: bool = True

    # Intent classification settings
    intent_classification_timeout: int = 10
    intent_fallback: str = "research_query"

    # Tool execution settings
    tool_execution_timeout: int = 30
    tool_max_retries: int = 2

    # Checkpoint settings
    checkpoint_enabled: bool = True
    checkpoint_ttl_seconds: int = 86400 * 7  # 7 days

    # CORS settings
    cors_origins: list[str] = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]
    
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string or JSON list."""
        if isinstance(v, str):
            # Try JSON first
            try:
                import json
                return json.loads(v)
            except (json.JSONDecodeError, ValueError):
                # If not JSON, treat as comma-separated string
                return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_config(self):
        """Validate all required configuration."""
        errors = []
        
        # Validate Supabase
        if not self.supabase_url:
            errors.append("SUPABASE_URL is required")
        elif not self.supabase_url.startswith("http"):
            errors.append("SUPABASE_URL must be a valid HTTP/HTTPS URL")
        
        if not self.supabase_service_role_key and not self.supabase_service_key:
            errors.append("Either SUPABASE_SERVICE_ROLE_KEY or SUPABASE_SERVICE_KEY must be set")
        
        # Validate service URLs
        if not self.llm_service_url:
            errors.append("LLM_SERVICE_URL is required")
        elif not self.llm_service_url.startswith("http"):
            errors.append("LLM_SERVICE_URL must be a valid HTTP/HTTPS URL")
        
        if not self.rag_service_url:
            errors.append("RAG_SERVICE_URL is required")
        elif not self.rag_service_url.startswith("http"):
            errors.append("RAG_SERVICE_URL must be a valid HTTP/HTTPS URL")
        
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
        
        return self

    @property
    def supabase_key(self) -> str:
        """Get Supabase service key (prefer service_role_key)."""
        return self.supabase_service_role_key or self.supabase_service_key or ""


settings = Settings()

