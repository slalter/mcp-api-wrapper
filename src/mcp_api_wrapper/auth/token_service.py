"""Token generation and validation service.

Auth is decoupled from MCP - this service handles API-level authentication.
Supports scoped tokens with TTL and rate limiting.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from pydantic import BaseModel, Field

from mcp_api_wrapper.config import Settings


class TokenClaims(BaseModel):
    """Claims embedded in an issued JWT."""

    sub: str = Field(description="Client identifier")
    scopes: list[str] = Field(default_factory=list)
    iss: str = Field(default="mcp-api-wrapper")
    iat: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    exp: datetime = Field(description="Expiration time")
    rate_limit_rpm: int = Field(default=60)
    rate_limit_rph: int = Field(default=1000)


class TokenService:
    """Generates and validates API access tokens (JWTs).

    This is the core of the decoupled auth model: tokens are issued
    through the MCP layer but validated at the API layer independently.
    """

    def __init__(self, settings: Settings) -> None:
        self._secret = settings.auth_secret_key
        self._issuer = settings.auth_issuer
        self._ttl_minutes = settings.auth_token_ttl_minutes
        self._algorithm = "HS256"
        # In-memory client registry (replace with DB / external auth in production)
        self._clients: dict[str, _ClientRecord] = {}

    def register_client(
        self,
        client_id: str,
        client_secret: str,
        allowed_scopes: list[str] | None = None,
        rate_limit_rpm: int = 60,
    ) -> None:
        """Register a client that can request tokens."""
        self._clients[client_id] = _ClientRecord(
            client_id=client_id,
            client_secret=client_secret,
            allowed_scopes=allowed_scopes or ["read"],
            rate_limit_rpm=rate_limit_rpm,
        )

    def issue_token(
        self,
        client_id: str,
        client_secret: str,
        requested_scopes: list[str] | None = None,
    ) -> tuple[str, TokenClaims]:
        """Authenticate a client and issue a scoped JWT.

        Returns (token_string, claims) on success.
        Raises ValueError on auth failure.
        """
        record = self._clients.get(client_id)
        if record is None or record.client_secret != client_secret:
            raise ValueError("Invalid client credentials")

        # Scope intersection: grant only what the client is allowed
        requested = set(requested_scopes or record.allowed_scopes)
        granted = sorted(requested & set(record.allowed_scopes))
        if not granted:
            raise ValueError(
                f"None of the requested scopes are allowed. "
                f"Allowed: {record.allowed_scopes}"
            )

        now = datetime.now(timezone.utc)
        claims = TokenClaims(
            sub=client_id,
            scopes=granted,
            iss=self._issuer,
            iat=now,
            exp=now + timedelta(minutes=self._ttl_minutes),
            rate_limit_rpm=record.rate_limit_rpm,
        )

        payload: dict[str, Any] = {
            "sub": claims.sub,
            "scopes": claims.scopes,
            "iss": claims.iss,
            "iat": claims.iat,
            "exp": claims.exp,
            "rate_limit_rpm": claims.rate_limit_rpm,
            "rate_limit_rph": claims.rate_limit_rph,
        }
        token = jwt.encode(payload, self._secret, algorithm=self._algorithm)
        return token, claims

    def validate_token(self, token: str) -> TokenClaims:
        """Validate a JWT and return its claims.

        Raises jwt.InvalidTokenError on failure.
        """
        payload = jwt.decode(
            token,
            self._secret,
            algorithms=[self._algorithm],
            issuer=self._issuer,
        )
        return TokenClaims(
            sub=payload["sub"],
            scopes=payload["scopes"],
            iss=payload["iss"],
            iat=datetime.fromtimestamp(payload["iat"], tz=timezone.utc),
            exp=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
            rate_limit_rpm=payload.get("rate_limit_rpm", 60),
            rate_limit_rph=payload.get("rate_limit_rph", 1000),
        )


class _ClientRecord(BaseModel):
    """Internal client registration record."""

    client_id: str
    client_secret: str
    allowed_scopes: list[str]
    rate_limit_rpm: int = 60
