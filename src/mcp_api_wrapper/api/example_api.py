"""Example FastAPI application that sits behind the MCP wrapper.

This demonstrates what a "wrapped" API looks like. In production,
this would be your actual API - the MCP server just provides
discovery, auth, and docs on top of it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Header
from pydantic import BaseModel, Field

from mcp_api_wrapper.auth.token_service import TokenClaims, TokenService
from mcp_api_wrapper.config import Settings, get_settings


def create_example_api(
    settings: Settings | None = None,
    token_service: TokenService | None = None,
) -> FastAPI:
    """Create the example API application."""
    settings = settings or get_settings()

    if token_service is None:
        token_service = TokenService(settings)

    app = FastAPI(
        title="Example Wrapped API",
        description="A sample API that is discovered and authenticated via the MCP wrapper",
        version=settings.api_version,
    )

    # Dependency: validate Bearer token
    async def get_current_client(
        authorization: Annotated[str, Header()],
    ) -> TokenClaims:
        if not authorization.startswith("Bearer "):
            raise HTTPException(401, "Missing Bearer token")
        token = authorization[7:]
        try:
            assert token_service is not None
            return token_service.validate_token(token)
        except Exception:
            raise HTTPException(401, "Invalid or expired token")

    # ── Public endpoints ───────────────────────────────────────────────

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": settings.api_version}

    @app.get("/docs/openapi.json")
    async def openapi_docs() -> dict[str, object]:
        schema = app.openapi()
        return schema  # type: ignore[return-value]

    # ── Protected endpoints ────────────────────────────────────────────

    @app.get("/users")
    async def list_users(
        client: Annotated[TokenClaims, Depends(get_current_client)],
    ) -> list[UserResponse]:
        """List all users (requires 'read' scope)."""
        if "read" not in client.scopes:
            raise HTTPException(403, "Insufficient scope: requires 'read'")
        return [
            UserResponse(
                id="user-1",
                name="Alice",
                email="alice@example.com",
                created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            ),
            UserResponse(
                id="user-2",
                name="Bob",
                email="bob@example.com",
                created_at=datetime(2025, 6, 15, tzinfo=timezone.utc),
            ),
        ]

    @app.get("/users/{user_id}")
    async def get_user(
        user_id: str,
        client: Annotated[TokenClaims, Depends(get_current_client)],
    ) -> UserResponse:
        """Get a specific user (requires 'read' scope)."""
        if "read" not in client.scopes:
            raise HTTPException(403, "Insufficient scope: requires 'read'")
        # Stub: return mock data
        return UserResponse(
            id=user_id,
            name="Alice",
            email="alice@example.com",
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )

    @app.post("/users")
    async def create_user(
        body: CreateUserRequest,
        client: Annotated[TokenClaims, Depends(get_current_client)],
    ) -> UserResponse:
        """Create a new user (requires 'write' scope)."""
        if "write" not in client.scopes:
            raise HTTPException(403, "Insufficient scope: requires 'write'")
        return UserResponse(
            id="user-new",
            name=body.name,
            email=body.email,
            created_at=datetime.now(timezone.utc),
        )

    return app


# ── Request/Response models ────────────────────────────────────────────────


class UserResponse(BaseModel):
    id: str
    name: str
    email: str
    created_at: datetime


class CreateUserRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(min_length=3, max_length=255)
