#!/usr/bin/env python3
"""Complete end-to-end example of the MCP-as-API-Wrapper pattern.

This script demonstrates the full flow:
1. Start an example API server
2. Create the MCP server with the 3 thin tools
3. Simulate a client that:
   a. Discovers API endpoints via MCP
   b. Authenticates via MCP to get a Bearer token
   c. Fetches documentation URL via MCP
   d. Makes direct API calls using the Bearer token (bypassing MCP)
4. Simulate the agent-to-agent evolution flow:
   a. Client requests a new endpoint via the message queue
   b. Backend agent "implements" the change
   c. Client re-fetches docs and sees the update

Run:
    python examples/full_flow.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone

import httpx
import uvicorn

from mcp_api_wrapper.api.example_api import create_example_api
from mcp_api_wrapper.auth.token_service import TokenService
from mcp_api_wrapper.config import Settings
from mcp_api_wrapper.queue.message_queue import InMemoryMessageQueue
from mcp_api_wrapper.schemas import (
    APIEvolutionRequest,
    APIEvolutionResponse,
    EvolutionRequestStatus,
    HTTPMethod,
)
from mcp_api_wrapper.rag.types import DocFormat
from mcp_api_wrapper.server.main import (
    _handle_get_auth_token,
    _handle_get_documentation,
    _handle_get_endpoints,
    _handle_search_documentation,
    _ServerState,
)


def banner(text: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


async def run_example() -> None:
    # ── Setup ──────────────────────────────────────────────────────────────
    settings = Settings(
        api_base_url="http://localhost:8100",
        api_version="1.0.0",
        auth_secret_key="example-secret-key-at-least-32-chars!",
        auth_token_ttl_minutes=60,
        mcp_server_port=8931,
    )

    # Create the shared state (same objects the MCP server uses)
    state = _ServerState(settings)

    # Register a demo client
    state.token_service.register_client(
        client_id="demo-agent",
        client_secret="demo-secret-123",
        allowed_scopes=["read", "write"],
        rate_limit_rpm=100,
    )

    # Pre-populate the registry with the example API's endpoints
    state.registry.register_endpoint(
        "/health", [HTTPMethod.GET], "Health check", auth_required=False
    )
    state.registry.register_endpoint(
        "/users", [HTTPMethod.GET, HTTPMethod.POST], "User management"
    )
    state.registry.register_endpoint(
        "/users/{user_id}", [HTTPMethod.GET], "Get single user"
    )

    # Create the example API app (sharing the same token service)
    api_app = create_example_api(settings, state.token_service)

    # Start API server in background
    config = uvicorn.Config(api_app, host="127.0.0.1", port=8100, log_level="warning")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    await asyncio.sleep(0.5)  # Wait for server startup

    try:
        # ── Phase 1: Discovery via MCP ─────────────────────────────────────
        banner("PHASE 1: Discover API Endpoints (via MCP)")

        result = await _handle_get_endpoints(state)
        endpoints_data = json.loads(result[0].text)
        print(f"API Base URL: {endpoints_data['api_base_url']}")
        print(f"API Version:  {endpoints_data['api_version']}")
        print(f"Endpoints ({len(endpoints_data['endpoints'])}):")
        for ep in endpoints_data["endpoints"]:
            methods = ", ".join(ep["methods"])
            auth = "🔒" if ep["auth_required"] else "🔓"
            print(f"  {auth} {methods:20s} {ep['path']:25s} {ep['summary']}")

        # ── Phase 2: Authenticate via MCP ──────────────────────────────────
        banner("PHASE 2: Get Auth Token (via MCP)")

        result = await _handle_get_auth_token(state, {
            "client_id": "demo-agent",
            "client_secret": "demo-secret-123",
            "scopes": ["read", "write"],
        })
        auth_data = json.loads(result[0].text)
        token = auth_data["token"]
        print(f"Token Type:   {auth_data['token_type']}")
        print(f"Scopes:       {auth_data['scopes']}")
        print(f"Expires:      {auth_data['expires_at']}")
        print(f"API Base:     {auth_data['api_base_url']}")
        print(f"Rate Limit:   {auth_data['rate_limit']['requests_per_minute']} rpm")
        print(f"Token:        {token[:20]}...{token[-10:]}")

        # ── Phase 3: Get Documentation URL via MCP ─────────────────────────
        banner("PHASE 3: Get Documentation URL (via MCP)")

        result = await _handle_get_documentation(state, {"format": "openapi"})
        docs_data = json.loads(result[0].text)
        print(f"Docs URL:     {docs_data['docs_url']}")
        print(f"Format:       {docs_data['format']}")
        print(f"Version:      {docs_data['version']}")

        # ── Phase 3.5: Search Documentation via RAG ─────────────────────
        banner("PHASE 3.5: Search Documentation (via MCP RAG)")

        # Ingest the OpenAPI spec into the RAG index
        if state.doc_index is not None:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{settings.api_base_url}/docs/openapi.json")
                spec_content = resp.text
            chunk_count = state.doc_index.ingest(
                spec_content, source="openapi.json", doc_format=DocFormat.OPENAPI
            )
            print(f"Indexed {chunk_count} chunks from OpenAPI spec")

            # Search for user-related documentation
            result = await _handle_search_documentation(
                state, {"query": "how to list users", "top_k": 3}
            )
            search_data = json.loads(result[0].text)
            print(f"\nSearch: '{search_data['query']}'")
            print(f"Total chunks indexed: {search_data['total_chunks_indexed']}")
            print(f"Results ({len(search_data['results'])}):")
            for r in search_data["results"]:
                preview = r["content"][:80].replace("\n", " ")
                print(f"  [{r['score']:.3f}] {r['section']:30s} {preview}...")
        else:
            print("RAG disabled, skipping documentation search")

        # ── Phase 4: Direct API Calls (bypassing MCP!) ────────────────────
        banner("PHASE 4: Direct API Calls (Bearer Token, NO MCP)")

        async with httpx.AsyncClient() as client:
            # Health check (no auth needed)
            print("→ GET /health (no auth)")
            resp = await client.get(f"{settings.api_base_url}/health")
            print(f"  Status: {resp.status_code}")
            print(f"  Body:   {resp.json()}")

            # List users (auth required)
            print("\n→ GET /users (Bearer token)")
            resp = await client.get(
                f"{settings.api_base_url}/users",
                headers={"Authorization": f"Bearer {token}"},
            )
            print(f"  Status: {resp.status_code}")
            users = resp.json()
            for u in users:
                print(f"  User:   {u['name']} ({u['email']})")

            # Create a user (auth required, write scope)
            print("\n→ POST /users (Bearer token, write scope)")
            resp = await client.post(
                f"{settings.api_base_url}/users",
                headers={"Authorization": f"Bearer {token}"},
                json={"name": "Charlie", "email": "charlie@example.com"},
            )
            print(f"  Status: {resp.status_code}")
            print(f"  Created: {resp.json()}")

            # Try without auth (should fail)
            print("\n→ GET /users (NO auth - should fail)")
            resp = await client.get(f"{settings.api_base_url}/users")
            print(f"  Status: {resp.status_code} (expected 422 - missing header)")

        # ── Phase 5: Dynamic API Evolution ─────────────────────────────────
        banner("PHASE 5: Agent-to-Agent API Evolution")

        queue = state.message_queue

        # Client agent requests a new endpoint
        request = APIEvolutionRequest(
            request_id="req-001",
            from_client="demo-agent",
            request_type="add_endpoint",
            description="Add GET /users/{id}/activity to retrieve user activity log",
            rationale="Need activity data for the dashboard feature",
            priority="high",
        )
        print(f"Client → Queue: Requesting '{request.description}'")
        await queue.publish("api-requests", request, "client_to_backend")
        print(f"  Request status: {queue.get_request_status('req-001')}")

        # Backend agent picks up and processes
        msg = await queue.subscribe("api-requests", timeout=5.0)
        assert msg is not None
        print(f"\nBackend Agent received: {msg.payload}")

        # Backend agent "implements" the change
        state.registry.register_endpoint(
            "/users/{user_id}/activity",
            [HTTPMethod.GET],
            "User activity log",
        )
        state.registry.set_version("1.1.0")

        # Backend agent responds
        response = APIEvolutionResponse(
            request_id="req-001",
            status=EvolutionRequestStatus.COMPLETED,
            message="Added GET /users/{id}/activity endpoint. Deployed as v1.1.0.",
            new_version="1.1.0",
            new_endpoints=["GET /users/{user_id}/activity"],
            completed_at=datetime.now(timezone.utc),
        )
        await queue.publish("api-responses", response, "backend_to_client")
        print(f"\nBackend Agent → Queue: '{response.message}'")
        print(f"  Request status: {queue.get_request_status('req-001')}")

        # Client re-fetches endpoints and sees the new one
        print("\nClient re-fetches endpoints via MCP...")
        result = await _handle_get_endpoints(state)
        updated = json.loads(result[0].text)
        print(f"  API Version: {updated['api_version']} (was 1.0.0)")
        print(f"  Endpoints ({len(updated['endpoints'])}):")
        for ep in updated["endpoints"]:
            methods = ", ".join(ep["methods"])
            new_badge = " ← NEW!" if ep["version_added"] == "1.1.0" else ""
            print(f"    {methods:20s} {ep['path']:30s}{new_badge}")

        # ── Done ───────────────────────────────────────────────────────────
        banner("COMPLETE: Full MCP-as-API-Wrapper Flow Demonstrated")
        print("Key takeaways:")
        print("  • MCP server exposed 4 tools (3 original + searchDocumentation)")
        print("  • searchDocumentation uses RAG to find relevant doc chunks")
        print("  • Auth was handled at the API layer, not MCP")
        print("  • Client made direct HTTP calls with Bearer token")
        print("  • API evolved dynamically via agent-to-agent queue")
        print("  • Client discovered new endpoints by re-calling MCP tools")
        print("  • Zero MCP reconnections throughout the entire flow")
        print()

    finally:
        server.should_exit = True
        await server_task


if __name__ == "__main__":
    asyncio.run(run_example())
