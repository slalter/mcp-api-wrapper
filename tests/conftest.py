"""Shared test fixtures."""

from __future__ import annotations

import pytest

from mcp_api_wrapper.api.registry import EndpointRegistry
from mcp_api_wrapper.auth.token_service import TokenService
from mcp_api_wrapper.config import Settings
from mcp_api_wrapper.queue.message_queue import InMemoryMessageQueue
from mcp_api_wrapper.schemas import HTTPMethod


@pytest.fixture
def settings() -> Settings:
    return Settings(
        api_base_url="http://test-api:8000",
        api_version="1.0.0",
        auth_secret_key="test-secret-key-do-not-use",
        auth_token_ttl_minutes=30,
    )


@pytest.fixture
def token_service(settings: Settings) -> TokenService:
    svc = TokenService(settings)
    svc.register_client(
        client_id="test-client",
        client_secret="test-secret",
        allowed_scopes=["read", "write", "admin"],
        rate_limit_rpm=100,
    )
    svc.register_client(
        client_id="readonly-client",
        client_secret="readonly-secret",
        allowed_scopes=["read"],
        rate_limit_rpm=30,
    )
    return svc


@pytest.fixture
def registry(settings: Settings) -> EndpointRegistry:
    reg = EndpointRegistry(settings)
    reg.register_endpoint("/users", [HTTPMethod.GET, HTTPMethod.POST], "User management")
    reg.register_endpoint("/users/{id}", [HTTPMethod.GET, HTTPMethod.PUT], "Single user")
    reg.register_endpoint("/health", [HTTPMethod.GET], "Health check", auth_required=False)
    return reg


@pytest.fixture
def message_queue() -> InMemoryMessageQueue:
    return InMemoryMessageQueue()
