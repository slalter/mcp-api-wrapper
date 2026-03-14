"""Tests for the thin MCP server (3 tools)."""

from __future__ import annotations

import json

import pytest

from mcp_api_wrapper.config import Settings
from mcp_api_wrapper.schemas import HTTPMethod
from mcp_api_wrapper.server.main import create_server


class TestMCPServerTools:
    @pytest.fixture
    def server_state(self, settings: Settings) -> tuple:
        server, state = create_server(settings)
        # Register a test client
        state.token_service.register_client(
            "test-client", "test-secret", ["read", "write"]
        )
        # Register some endpoints
        state.registry.register_endpoint(
            "/users", [HTTPMethod.GET, HTTPMethod.POST], "User management"
        )
        state.registry.register_endpoint("/health", [HTTPMethod.GET], "Health", auth_required=False)
        return server, state

    def test_create_server_returns_server_and_state(self, server_state: tuple) -> None:
        server, state = server_state
        assert server is not None
        assert state.registry is not None
        assert state.token_service is not None
        assert state.message_queue is not None

    @pytest.mark.asyncio
    async def test_get_api_endpoints(self, settings: Settings) -> None:
        _, state = create_server(settings)
        state.registry.register_endpoint("/users", [HTTPMethod.GET], "Users")

        from mcp_api_wrapper.server.main import _handle_get_endpoints

        result = await _handle_get_endpoints(state)
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["api_base_url"] == "http://test-api:8000"
        assert data["api_version"] == "1.0.0"
        assert len(data["endpoints"]) == 1
        assert data["endpoints"][0]["path"] == "/users"

    @pytest.mark.asyncio
    async def test_get_auth_token_success(self, settings: Settings) -> None:
        _, state = create_server(settings)
        state.token_service.register_client("c1", "s1", ["read"])

        from mcp_api_wrapper.server.main import _handle_get_auth_token

        result = await _handle_get_auth_token(
            state, {"client_id": "c1", "client_secret": "s1"}
        )
        data = json.loads(result[0].text)
        assert "token" in data
        assert data["token_type"] == "Bearer"
        assert "read" in data["scopes"]

    @pytest.mark.asyncio
    async def test_get_auth_token_failure(self, settings: Settings) -> None:
        _, state = create_server(settings)

        from mcp_api_wrapper.server.main import _handle_get_auth_token

        result = await _handle_get_auth_token(
            state, {"client_id": "bad", "client_secret": "bad"}
        )
        data = json.loads(result[0].text)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_get_documentation(self, settings: Settings) -> None:
        _, state = create_server(settings)

        from mcp_api_wrapper.server.main import _handle_get_documentation

        result = await _handle_get_documentation(state, {"format": "openapi"})
        data = json.loads(result[0].text)
        assert data["docs_url"] == "http://test-api:8000/docs/openapi.json"
        assert data["format"] == "openapi"
        assert data["version"] == "1.0.0"

    @pytest.mark.asyncio
    async def test_get_documentation_markdown(self, settings: Settings) -> None:
        _, state = create_server(settings)

        from mcp_api_wrapper.server.main import _handle_get_documentation

        result = await _handle_get_documentation(state, {"format": "markdown"})
        data = json.loads(result[0].text)
        assert data["docs_url"] == "http://test-api:8000/docs/api.md"
        assert data["format"] == "markdown"

    @pytest.mark.asyncio
    async def test_get_documentation_invalid_format_defaults(self, settings: Settings) -> None:
        _, state = create_server(settings)

        from mcp_api_wrapper.server.main import _handle_get_documentation

        result = await _handle_get_documentation(state, {"format": "pdf"})
        data = json.loads(result[0].text)
        assert data["format"] == "openapi"  # defaults to openapi
