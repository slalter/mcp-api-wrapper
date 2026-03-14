"""Thin MCP server with exactly 3 tools.

This is the stable layer that never changes. The 3 tools provide:
1. getAPIEndpoints - Discover what the API can do
2. getAuthToken - Get authenticated for API access
3. getDocumentation - Get full API docs

All real work happens through the API that sits behind these tools.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from mcp_api_wrapper.api.registry import EndpointRegistry
from mcp_api_wrapper.auth.token_service import TokenService
from mcp_api_wrapper.config import Settings, get_settings
from mcp_api_wrapper.queue.message_queue import InMemoryMessageQueue
from mcp_api_wrapper.schemas import (
    APIEvolutionRequest,
    AuthRequest,
    DocumentationResponse,
)


def create_server(settings: Settings | None = None) -> tuple[Server, _ServerState]:
    """Create the MCP server with the 3 thin tools.

    Returns (server, state) so the caller can inject dependencies.
    """
    settings = settings or get_settings()
    state = _ServerState(settings)
    server = Server(settings.mcp_server_name)

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="getAPIEndpoints",
                description=(
                    "Discover all available API endpoints, their methods, "
                    "and the current API version. Call this to understand "
                    "what the API can do before making direct API calls."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="getAuthToken",
                description=(
                    "Authenticate and receive a scoped Bearer token for "
                    "direct API access. The token is independent of the MCP "
                    "session and can be used for direct HTTP calls to the API."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "client_id": {
                            "type": "string",
                            "description": "Your client identifier",
                        },
                        "client_secret": {
                            "type": "string",
                            "description": "Your client secret",
                        },
                        "scopes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Requested permission scopes (e.g. ['read', 'write'])",
                            "default": [],
                        },
                    },
                    "required": ["client_id", "client_secret"],
                    "additionalProperties": False,
                },
            ),
            Tool(
                name="getDocumentation",
                description=(
                    "Get the URL to download full API documentation. "
                    "The docs always reflect the latest deployed API version. "
                    "Use the returned auth_header when fetching the docs URL."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "format": {
                            "type": "string",
                            "enum": ["openapi", "markdown", "html"],
                            "description": "Preferred documentation format",
                            "default": "openapi",
                        },
                    },
                    "additionalProperties": False,
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, object]) -> list[TextContent]:
        if name == "getAPIEndpoints":
            return await _handle_get_endpoints(state)
        elif name == "getAuthToken":
            return await _handle_get_auth_token(state, arguments)
        elif name == "getDocumentation":
            return await _handle_get_documentation(state, arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    return server, state


async def _handle_get_endpoints(state: _ServerState) -> list[TextContent]:
    """Handle getAPIEndpoints tool call."""
    response = state.registry.get_all()
    return [TextContent(type="text", text=response.model_dump_json(indent=2))]


async def _handle_get_auth_token(
    state: _ServerState, arguments: dict[str, object]
) -> list[TextContent]:
    """Handle getAuthToken tool call."""
    try:
        req = AuthRequest(
            client_id=str(arguments.get("client_id", "")),
            client_secret=str(arguments.get("client_secret", "")),
            scopes=[str(s) for s in arguments.get("scopes", [])]  # type: ignore[union-attr]
            if isinstance(arguments.get("scopes"), list)
            else [],
        )
        token, claims = state.token_service.issue_token(
            client_id=req.client_id,
            client_secret=req.client_secret,
            requested_scopes=req.scopes if req.scopes else None,
        )
        result = {
            "token": token,
            "token_type": "Bearer",
            "scopes": claims.scopes,
            "expires_at": claims.exp.isoformat(),
            "api_base_url": state.settings.api_base_url,
            "rate_limit": {
                "requests_per_minute": claims.rate_limit_rpm,
                "requests_per_hour": claims.rate_limit_rph,
            },
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except ValueError as e:
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def _handle_get_documentation(
    state: _ServerState, arguments: dict[str, object]
) -> list[TextContent]:
    """Handle getDocumentation tool call."""
    fmt = str(arguments.get("format", "openapi"))
    if fmt not in ("openapi", "markdown", "html"):
        fmt = "openapi"

    docs_path_map = {
        "openapi": "/docs/openapi.json",
        "markdown": "/docs/api.md",
        "html": "/docs",
    }
    docs_path = docs_path_map[fmt]

    response = DocumentationResponse(
        docs_url=f"{state.settings.api_base_url}{docs_path}",
        format=fmt,  # type: ignore[arg-type]
        version=state.registry.version,
        updated_at=datetime.now(timezone.utc),
    )
    return [TextContent(type="text", text=response.model_dump_json(indent=2))]


class _ServerState:
    """Holds shared state for the MCP server tools."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.registry = EndpointRegistry(settings)
        self.token_service = TokenService(settings)
        self.message_queue = InMemoryMessageQueue()


async def _run_server() -> None:
    """Run the MCP server over stdio."""
    settings = get_settings()
    server, _state = create_server(settings)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    """Entry point for the MCP server."""
    import asyncio

    asyncio.run(_run_server())


if __name__ == "__main__":
    main()
