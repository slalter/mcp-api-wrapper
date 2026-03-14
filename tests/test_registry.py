"""Tests for the API endpoint registry."""

from __future__ import annotations

from mcp_api_wrapper.api.registry import EndpointRegistry
from mcp_api_wrapper.config import Settings
from mcp_api_wrapper.schemas import HTTPMethod


class TestEndpointRegistry:
    def test_register_and_list(self, registry: EndpointRegistry) -> None:
        result = registry.get_all()
        assert len(result.endpoints) == 3
        paths = [e.path for e in result.endpoints]
        assert "/users" in paths
        assert "/users/{id}" in paths
        assert "/health" in paths

    def test_api_base_url(self, registry: EndpointRegistry) -> None:
        result = registry.get_all()
        assert result.api_base_url == "http://test-api:8000"

    def test_version_tracking(self, registry: EndpointRegistry) -> None:
        assert registry.version == "1.0.0"
        registry.set_version("2.0.0")
        assert registry.version == "2.0.0"
        result = registry.get_all()
        assert result.api_version == "2.0.0"

    def test_remove_endpoint(self, registry: EndpointRegistry) -> None:
        assert registry.remove_endpoint("/health")
        result = registry.get_all()
        assert len(result.endpoints) == 2
        assert "/health" not in [e.path for e in result.endpoints]

    def test_remove_nonexistent(self, registry: EndpointRegistry) -> None:
        assert not registry.remove_endpoint("/nonexistent")

    def test_auth_required_flag(self, registry: EndpointRegistry) -> None:
        result = registry.get_all()
        health = next(e for e in result.endpoints if e.path == "/health")
        users = next(e for e in result.endpoints if e.path == "/users")
        assert not health.auth_required
        assert users.auth_required

    def test_docs_url(self, registry: EndpointRegistry) -> None:
        result = registry.get_all()
        assert result.docs_url == "http://test-api:8000/docs/openapi.json"

    def test_refresh_from_openapi(self, settings: Settings) -> None:
        reg = EndpointRegistry(settings)
        openapi_spec: dict[str, object] = {
            "info": {"title": "Test", "version": "3.0.0"},
            "paths": {
                "/items": {
                    "get": {"summary": "List items"},
                    "post": {"summary": "Create item"},
                },
                "/items/{id}": {
                    "get": {"summary": "Get item"},
                    "delete": {"summary": "Delete item"},
                },
            },
        }
        reg.refresh_from_openapi(openapi_spec)
        result = reg.get_all()
        assert reg.version == "3.0.0"
        assert len(result.endpoints) == 2
        items_ep = next(e for e in result.endpoints if e.path == "/items")
        assert HTTPMethod.GET in items_ep.methods
        assert HTTPMethod.POST in items_ep.methods
