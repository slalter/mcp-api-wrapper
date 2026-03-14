"""Shared Pydantic schemas for the MCP-as-API-Wrapper system."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ── API Discovery ──────────────────────────────────────────────────────────────


class HTTPMethod(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


class EndpointInfo(BaseModel):
    """Single API endpoint descriptor."""

    path: str = Field(description="URL path, e.g. /users/{id}")
    methods: list[HTTPMethod] = Field(description="Supported HTTP methods")
    summary: str = Field(default="", description="Short description")
    auth_required: bool = Field(default=True)
    version_added: str = Field(default="1.0.0", description="API version this was added in")


class APIEndpointsResponse(BaseModel):
    """Response from getAPIEndpoints tool."""

    api_base_url: str = Field(description="Base URL for all API calls")
    api_version: str = Field(description="Current API version")
    endpoints: list[EndpointInfo] = Field(description="Available endpoints")
    docs_url: str = Field(description="URL to full documentation")
    updated_at: datetime = Field(description="When endpoint list was last updated")


# ── Authentication ─────────────────────────────────────────────────────────────


class AuthRequest(BaseModel):
    """Request to getAuthToken tool."""

    client_id: str = Field(description="Client identifier")
    client_secret: str = Field(default="", description="Client secret (if applicable)")
    scopes: list[str] = Field(default_factory=list, description="Requested permission scopes")
    auth_method: Literal["api_key", "oauth2", "certificate"] = Field(
        default="api_key", description="Authentication method to use"
    )


class RateLimitInfo(BaseModel):
    """Rate limiting details for an issued token."""

    requests_per_minute: int = Field(default=60)
    requests_per_hour: int = Field(default=1000)
    burst_limit: int = Field(default=10)


class AuthTokenResponse(BaseModel):
    """Response from getAuthToken tool."""

    token: str = Field(description="Bearer token for API access")
    token_type: Literal["Bearer"] = "Bearer"
    scopes: list[str] = Field(description="Granted permission scopes")
    expires_at: datetime = Field(description="Token expiration time")
    api_base_url: str = Field(description="API base URL to use with this token")
    rate_limit: RateLimitInfo = Field(default_factory=RateLimitInfo)


# ── Documentation ──────────────────────────────────────────────────────────────


class DocumentationResponse(BaseModel):
    """Response from getDocumentation tool."""

    docs_url: str = Field(description="URL to download full API documentation")
    format: Literal["openapi", "markdown", "html"] = Field(
        default="openapi", description="Documentation format"
    )
    version: str = Field(description="Documentation version (matches API version)")
    auth_header: str = Field(
        default="",
        description="Authorization header value to include when fetching docs (if needed)",
    )
    updated_at: datetime = Field(description="When docs were last updated")
    changelog_url: str = Field(default="", description="URL to changelog / diff from last version")


# ── Documentation Search (RAG) ────────────────────────────────────────────────


class SearchDocumentationRequest(BaseModel):
    """Request to searchDocumentation tool."""

    query: str = Field(description="Natural language search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Max results to return")
    section: str | None = Field(
        default=None,
        description="Filter results to a specific section (e.g. 'paths./users.get')",
    )


class DocSearchResult(BaseModel):
    """A single documentation search result."""

    content: str = Field(description="Matched documentation chunk")
    score: float = Field(description="Relevance score (0-1)")
    source: str = Field(description="Source document identifier")
    section: str = Field(description="Section within the document")


class SearchDocumentationResponse(BaseModel):
    """Response from searchDocumentation tool."""

    results: list[DocSearchResult] = Field(description="Ranked search results")
    total_chunks_indexed: int = Field(description="Total chunks in the index")
    query: str = Field(description="The original search query")


# ── Message Queue (Agent-to-Agent) ─────────────────────────────────────────────


class EvolutionRequestStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"


class APIEvolutionRequest(BaseModel):
    """Client agent requests an API change."""

    request_id: str = Field(description="Unique request identifier")
    from_client: str = Field(description="Client agent identifier")
    request_type: Literal["add_endpoint", "modify_endpoint", "add_field", "other"] = Field(
        description="Type of API change requested"
    )
    description: str = Field(description="Human-readable description of what's needed")
    rationale: str = Field(default="", description="Why this change is needed")
    priority: Literal["low", "medium", "high", "critical"] = Field(default="medium")
    suggested_spec: dict[str, object] | None = Field(
        default=None, description="Optional: suggested endpoint spec (OpenAPI fragment)"
    )


class APIEvolutionResponse(BaseModel):
    """Backend agent responds to an API change request."""

    request_id: str = Field(description="Original request identifier")
    status: EvolutionRequestStatus
    message: str = Field(description="Status message from backend agent")
    new_version: str | None = Field(
        default=None, description="New API version if changes were deployed"
    )
    new_endpoints: list[str] | None = Field(
        default=None, description="New endpoint paths if any were added"
    )
    completed_at: datetime | None = None
