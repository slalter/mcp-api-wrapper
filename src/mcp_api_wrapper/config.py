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

    model_config = {"env_prefix": "MCP_WRAPPER_"}


def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
