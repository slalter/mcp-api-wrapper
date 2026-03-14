"""Configuration for the MCP-as-API-Wrapper system."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings, loaded from environment variables."""

    # API configuration
    api_base_url: str = "http://localhost:8000"
    api_version: str = "1.0.0"
    api_docs_path: str = "/docs/openapi.json"

    # Auth configuration
    auth_secret_key: str = "dev-secret-change-in-production"
    auth_token_ttl_minutes: int = 60
    auth_issuer: str = "mcp-api-wrapper"

    # MCP server configuration
    mcp_server_name: str = "api-wrapper"
    mcp_server_host: str = "0.0.0.0"
    mcp_server_port: int = 8931

    # Queue configuration
    queue_backend: str = "memory"  # "memory" | "redis"
    redis_url: str = "redis://localhost:6379/0"

    # RAG configuration
    rag_enabled: bool = True
    rag_embedding_provider: str = "hash"  # "hash" | "openai"
    rag_openai_api_key: str = ""
    rag_max_chunk_size: int = 1500
    rag_chunk_overlap: int = 200
    rag_default_top_k: int = 5
    rag_min_score: float = 0.0

    model_config = {"env_prefix": "MCP_WRAPPER_"}


def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
