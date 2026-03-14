"""Integration tests: wire MCP server, API server, and auth together.

These tests exercise the full flow end-to-end, not just individual units.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import httpx
import pytest
import uvicorn

from mcp_api_wrapper.api.example_api import create_example_api
from mcp_api_wrapper.config import Settings
from mcp_api_wrapper.schemas import (
    APIEvolutionRequest,
    APIEvolutionResponse,
    EvolutionRequestStatus,
    HTTPMethod,
)
from mcp_api_wrapper.server.main import (
    _handle_get_auth_token,
    _handle_get_documentation,
    _handle_get_endpoints,
    _ServerState,
)


@pytest.fixture
def integration_settings() -> Settings:
    return Settings(
        api_base_url="http://127.0.0.1:8199",
        api_version="1.0.0",
        auth_secret_key="integration-test-secret-key-32chars!",
        auth_token_ttl_minutes=5,
    )


@pytest.fixture
def server_state(integration_settings: Settings) -> _ServerState:
    state = _ServerState(integration_settings)
    state.token_service.register_client(
        "test-agent", "test-secret", ["read", "write"], rate_limit_rpm=100
    )
    state.token_service.register_client(
        "readonly-agent", "readonly-secret", ["read"], rate_limit_rpm=50
    )
    state.registry.register_endpoint(
        "/health", [HTTPMethod.GET], "Health check", auth_required=False
    )
    state.registry.register_endpoint(
        "/users", [HTTPMethod.GET, HTTPMethod.POST], "User management"
    )
    state.registry.register_endpoint(
        "/users/{user_id}", [HTTPMethod.GET], "Get user"
    )
    return state


@pytest.fixture
async def running_api(
    integration_settings: Settings, server_state: _ServerState
) -> asyncio.Task[None]:  # type: ignore[type-arg]
    """Start the example API server in the background."""
    app = create_example_api(integration_settings, server_state.token_service)
    config = uvicorn.Config(app, host="127.0.0.1", port=8199, log_level="error")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.5)
    yield task  # type: ignore[misc]
    server.should_exit = True
    await task


class TestFullFlow:
    """Test the complete MCP → Auth → Direct API flow."""

    @pytest.mark.asyncio
    async def test_discover_then_auth_then_call_api(
        self,
        server_state: _ServerState,
        running_api: asyncio.Task[None],  # type: ignore[type-arg]
    ) -> None:
        """Full flow: discover endpoints, authenticate, call API directly."""
        # Step 1: Discover endpoints via MCP
        result = await _handle_get_endpoints(server_state)
        endpoints = json.loads(result[0].text)
        assert endpoints["api_version"] == "1.0.0"
        assert len(endpoints["endpoints"]) == 3
        api_base = endpoints["api_base_url"]

        # Step 2: Authenticate via MCP
        result = await _handle_get_auth_token(server_state, {
            "client_id": "test-agent",
            "client_secret": "test-secret",
            "scopes": ["read"],
        })
        auth = json.loads(result[0].text)
        assert "token" in auth
        assert "read" in auth["scopes"]
        token = auth["token"]

        # Step 3: Call API directly with Bearer token
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{api_base}/users",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp.status_code == 200
            users = resp.json()
            assert len(users) >= 1
            assert users[0]["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_scope_enforcement_on_api(
        self,
        server_state: _ServerState,
        running_api: asyncio.Task[None],  # type: ignore[type-arg]
    ) -> None:
        """Read-only token cannot write."""
        # Get read-only token
        result = await _handle_get_auth_token(server_state, {
            "client_id": "readonly-agent",
            "client_secret": "readonly-secret",
        })
        auth = json.loads(result[0].text)
        token = auth["token"]
        assert auth["scopes"] == ["read"]

        async with httpx.AsyncClient() as client:
            # Read works
            resp = await client.get(
                "http://127.0.0.1:8199/users",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp.status_code == 200

            # Write fails (403)
            resp = await client.post(
                "http://127.0.0.1:8199/users",
                headers={"Authorization": f"Bearer {token}"},
                json={"name": "Hacker", "email": "hack@evil.com"},
            )
            assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_unauthenticated_api_call_fails(
        self,
        running_api: asyncio.Task[None],  # type: ignore[type-arg]
    ) -> None:
        """API calls without a token are rejected."""
        async with httpx.AsyncClient() as client:
            # No auth header at all
            resp = await client.get("http://127.0.0.1:8199/users")
            assert resp.status_code == 422  # Missing required header

            # Invalid token
            resp = await client.get(
                "http://127.0.0.1:8199/users",
                headers={"Authorization": "Bearer invalid-garbage-token"},
            )
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_health_endpoint_no_auth(
        self,
        running_api: asyncio.Task[None],  # type: ignore[type-arg]
    ) -> None:
        """Health endpoint works without authentication."""
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://127.0.0.1:8199/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_docs_url_matches_api(
        self,
        server_state: _ServerState,
        running_api: asyncio.Task[None],  # type: ignore[type-arg]
    ) -> None:
        """Documentation URL from MCP tool actually returns valid OpenAPI."""
        result = await _handle_get_documentation(server_state, {"format": "openapi"})
        docs = json.loads(result[0].text)

        async with httpx.AsyncClient() as client:
            resp = await client.get(docs["docs_url"])
            assert resp.status_code == 200
            openapi = resp.json()
            assert "openapi" in openapi
            assert "paths" in openapi


class TestDynamicEvolution:
    """Test the agent-to-agent API evolution flow."""

    @pytest.mark.asyncio
    async def test_request_new_endpoint_and_rediscover(
        self,
        server_state: _ServerState,
    ) -> None:
        """Client requests new endpoint, backend implements, client rediscovers."""
        queue = server_state.message_queue

        # Initial state: 3 endpoints
        result = await _handle_get_endpoints(server_state)
        before = json.loads(result[0].text)
        assert len(before["endpoints"]) == 3
        assert before["api_version"] == "1.0.0"

        # Client requests new endpoint
        request = APIEvolutionRequest(
            request_id="int-test-001",
            from_client="test-agent",
            request_type="add_endpoint",
            description="Add GET /users/{id}/activity",
            priority="high",
        )
        await queue.publish("requests", request, "client_to_backend")

        # Backend agent processes
        msg = await queue.subscribe("requests", timeout=2.0)
        assert msg is not None
        assert isinstance(msg.payload, APIEvolutionRequest)

        # Backend implements the change
        server_state.registry.register_endpoint(
            "/users/{user_id}/activity",
            [HTTPMethod.GET],
            "User activity log",
        )
        server_state.registry.set_version("1.1.0")

        # Backend responds
        response = APIEvolutionResponse(
            request_id="int-test-001",
            status=EvolutionRequestStatus.COMPLETED,
            message="Deployed",
            new_version="1.1.0",
            new_endpoints=["GET /users/{user_id}/activity"],
            completed_at=datetime.now(timezone.utc),
        )
        await queue.publish("responses", response, "backend_to_client")

        # Client rediscovers: should see 4 endpoints, version 1.1.0
        result = await _handle_get_endpoints(server_state)
        after = json.loads(result[0].text)
        assert len(after["endpoints"]) == 4
        assert after["api_version"] == "1.1.0"
        paths = [e["path"] for e in after["endpoints"]]
        assert "/users/{user_id}/activity" in paths

    @pytest.mark.asyncio
    async def test_rejected_evolution_request(
        self,
        server_state: _ServerState,
    ) -> None:
        """Backend rejects a request; endpoint count doesn't change."""
        queue = server_state.message_queue

        result = await _handle_get_endpoints(server_state)
        before_count = len(json.loads(result[0].text)["endpoints"])

        # Client requests something unreasonable
        request = APIEvolutionRequest(
            request_id="int-test-002",
            from_client="test-agent",
            request_type="other",
            description="Add a nuclear launch endpoint",
            priority="critical",
        )
        await queue.publish("requests", request, "client_to_backend")

        # Backend rejects
        msg = await queue.subscribe("requests", timeout=2.0)
        assert msg is not None

        response = APIEvolutionResponse(
            request_id="int-test-002",
            status=EvolutionRequestStatus.REJECTED,
            message="This request violates safety policies.",
        )
        await queue.publish("responses", response, "backend_to_client")

        assert queue.get_request_status("int-test-002") == EvolutionRequestStatus.REJECTED

        # Endpoints unchanged
        result = await _handle_get_endpoints(server_state)
        after_count = len(json.loads(result[0].text)["endpoints"])
        assert after_count == before_count
