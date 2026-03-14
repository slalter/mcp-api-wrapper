"""API endpoint registry.

Tracks the current API surface so the MCP tools can report
what endpoints are available. The registry is updated when
the backend agent deploys changes.
"""

from __future__ import annotations

from datetime import datetime, timezone

from mcp_api_wrapper.config import Settings
from mcp_api_wrapper.schemas import APIEndpointsResponse, EndpointInfo, HTTPMethod


class EndpointRegistry:
    """Maintains the current set of API endpoints.

    This is the source of truth for getAPIEndpoints.
    Updated by the API layer on startup and by the backend agent
    when new endpoints are deployed.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._endpoints: list[EndpointInfo] = []
        self._version = settings.api_version
        self._updated_at = datetime.now(timezone.utc)

    @property
    def version(self) -> str:
        return self._version

    def register_endpoint(
        self,
        path: str,
        methods: list[HTTPMethod],
        summary: str = "",
        auth_required: bool = True,
    ) -> None:
        """Register a new API endpoint."""
        self._endpoints.append(
            EndpointInfo(
                path=path,
                methods=methods,
                summary=summary,
                auth_required=auth_required,
                version_added=self._version,
            )
        )
        self._updated_at = datetime.now(timezone.utc)

    def remove_endpoint(self, path: str) -> bool:
        """Remove an endpoint by path. Returns True if found."""
        before = len(self._endpoints)
        self._endpoints = [e for e in self._endpoints if e.path != path]
        if len(self._endpoints) < before:
            self._updated_at = datetime.now(timezone.utc)
            return True
        return False

    def set_version(self, version: str) -> None:
        """Update the API version (e.g. after backend agent deploys)."""
        self._version = version
        self._updated_at = datetime.now(timezone.utc)

    def get_all(self) -> APIEndpointsResponse:
        """Get the full endpoint listing."""
        return APIEndpointsResponse(
            api_base_url=self._settings.api_base_url,
            api_version=self._version,
            endpoints=list(self._endpoints),
            docs_url=f"{self._settings.api_base_url}{self._settings.api_docs_path}",
            updated_at=self._updated_at,
        )

    def refresh_from_openapi(self, openapi_spec: dict[str, object]) -> None:
        """Rebuild registry from an OpenAPI spec dict.

        Called when the API server restarts or the backend agent
        deploys a new version.
        """
        self._endpoints.clear()
        paths = openapi_spec.get("paths", {})
        if not isinstance(paths, dict):
            return

        info = openapi_spec.get("info", {})
        if isinstance(info, dict):
            version = info.get("version")
            if isinstance(version, str):
                self._version = version

        for path, methods_obj in paths.items():
            if not isinstance(methods_obj, dict):
                continue
            methods: list[HTTPMethod] = []
            summary = ""
            for method_str, detail in methods_obj.items():
                method_upper = method_str.upper()
                if method_upper in HTTPMethod.__members__:
                    methods.append(HTTPMethod(method_upper))
                    if isinstance(detail, dict) and not summary:
                        s = detail.get("summary")
                        if isinstance(s, str):
                            summary = s
            if methods:
                self.register_endpoint(
                    path=str(path),
                    methods=methods,
                    summary=summary,
                )
        self._updated_at = datetime.now(timezone.utc)
