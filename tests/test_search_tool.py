"""Tests for the searchDocumentation MCP tool integration."""

from __future__ import annotations

import json

import pytest

from mcp_api_wrapper.config import Settings
from mcp_api_wrapper.rag.doc_index import DocumentationIndex
from mcp_api_wrapper.rag.embeddings import HashEmbeddingProvider
from mcp_api_wrapper.rag.types import DocFormat
from mcp_api_wrapper.server.main import (
    _ServerState,
    _handle_search_documentation,
    create_server,
)


def _make_settings(**overrides: object) -> Settings:
    defaults = {
        "api_base_url": "http://test-api:8000",
        "auth_secret_key": "test-secret-key-at-least-32-chars!!",
        "rag_enabled": True,
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def state_with_index() -> _ServerState:
    """Server state with a pre-populated doc index."""
    settings = _make_settings()
    state = _ServerState(settings)
    assert state.doc_index is not None

    spec = json.dumps({
        "openapi": "3.0.0",
        "info": {"title": "Test API", "version": "1.0.0"},
        "paths": {
            "/users": {
                "get": {
                    "summary": "List all users",
                    "description": "Returns a paginated list of users",
                    "responses": {"200": {"description": "Success"}},
                },
                "post": {
                    "summary": "Create a user",
                    "responses": {"201": {"description": "Created"}},
                },
            },
            "/health": {
                "get": {
                    "summary": "Health check",
                    "responses": {"200": {"description": "OK"}},
                },
            },
        },
    })
    state.doc_index.ingest(spec, source="api.json", doc_format=DocFormat.OPENAPI)
    return state


class TestSearchDocumentationHandler:
    @pytest.mark.asyncio
    async def test_basic_search(self, state_with_index: _ServerState) -> None:
        result = await _handle_search_documentation(
            state_with_index, {"query": "list users"}
        )
        data = json.loads(result[0].text)
        assert "results" in data
        assert data["query"] == "list users"
        assert data["total_chunks_indexed"] > 0
        assert len(data["results"]) > 0

    @pytest.mark.asyncio
    async def test_rag_disabled_returns_error(self) -> None:
        settings = _make_settings(rag_enabled=False)
        state = _ServerState(settings)
        result = await _handle_search_documentation(state, {"query": "test"})
        data = json.loads(result[0].text)
        assert "error" in data
        assert "disabled" in data["error"]

    @pytest.mark.asyncio
    async def test_empty_index_returns_error(self) -> None:
        settings = _make_settings(rag_enabled=True)
        state = _ServerState(settings)
        # doc_index exists but has nothing ingested — clear it to be sure
        assert state.doc_index is not None
        state.doc_index.clear()
        result = await _handle_search_documentation(state, {"query": "test"})
        data = json.loads(result[0].text)
        # With empty index, search returns empty results (not an error)
        assert data["results"] == []

    @pytest.mark.asyncio
    async def test_section_filter(self, state_with_index: _ServerState) -> None:
        result = await _handle_search_documentation(
            state_with_index,
            {"query": "users", "section": "paths./users.get"},
        )
        data = json.loads(result[0].text)
        for r in data["results"]:
            assert r["section"] == "paths./users.get"

    @pytest.mark.asyncio
    async def test_top_k_clamping(self, state_with_index: _ServerState) -> None:
        # top_k=1 should return at most 1
        result = await _handle_search_documentation(
            state_with_index, {"query": "users", "top_k": 1}
        )
        data = json.loads(result[0].text)
        assert len(data["results"]) <= 1

    @pytest.mark.asyncio
    async def test_top_k_max_clamped(self, state_with_index: _ServerState) -> None:
        # top_k=100 should be clamped to 20
        result = await _handle_search_documentation(
            state_with_index, {"query": "users", "top_k": 100}
        )
        data = json.loads(result[0].text)
        assert len(data["results"]) <= 20

    @pytest.mark.asyncio
    async def test_empty_query_returns_error(self, state_with_index: _ServerState) -> None:
        result = await _handle_search_documentation(
            state_with_index, {"query": ""}
        )
        data = json.loads(result[0].text)
        assert "error" in data
        assert "required" in data["error"]

    @pytest.mark.asyncio
    async def test_result_shape(self, state_with_index: _ServerState) -> None:
        result = await _handle_search_documentation(
            state_with_index, {"query": "health check"}
        )
        data = json.loads(result[0].text)
        for r in data["results"]:
            assert "content" in r
            assert "score" in r
            assert "source" in r
            assert "section" in r
            assert isinstance(r["score"], float)


class TestToolListing:
    def test_doc_index_created_when_rag_enabled(self) -> None:
        settings = _make_settings(rag_enabled=True)
        _server, state = create_server(settings)
        assert state.doc_index is not None

    def test_doc_index_none_when_rag_disabled(self) -> None:
        settings = _make_settings(rag_enabled=False)
        _server, state = create_server(settings)
        assert state.doc_index is None
