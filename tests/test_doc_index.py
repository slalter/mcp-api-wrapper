"""Tests for DocumentationIndex orchestrator."""

from __future__ import annotations

import json

from mcp_api_wrapper.rag.doc_index import DocumentationIndex
from mcp_api_wrapper.rag.embeddings import HashEmbeddingProvider
from mcp_api_wrapper.rag.types import DocFormat


class TestDocumentationIndex:
    def test_ingest_and_search(self) -> None:
        idx = DocumentationIndex(embedder=HashEmbeddingProvider())
        count = idx.ingest(
            "# Users\nManage user accounts.\n\n# Auth\nAuthentication endpoints.\n",
            source="docs.md",
            doc_format=DocFormat.MARKDOWN,
        )
        assert count == 2
        assert idx.chunk_count == 2

        # Hash embeddings are deterministic but not semantically meaningful,
        # so we just verify the search returns results (any score)
        hits = idx.search("user accounts", min_score=-1.0)
        assert len(hits) > 0

    def test_openapi_ingestion(self) -> None:
        spec = json.dumps({
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0.0"},
            "paths": {
                "/users": {
                    "get": {"summary": "List users", "responses": {"200": {"description": "OK"}}},
                },
            },
        })
        idx = DocumentationIndex(embedder=HashEmbeddingProvider())
        count = idx.ingest(spec, source="api.json", doc_format=DocFormat.OPENAPI)
        # info chunk + 1 path chunk
        assert count == 2
        assert idx.chunk_count == 2

    def test_search_empty_index(self) -> None:
        idx = DocumentationIndex(embedder=HashEmbeddingProvider())
        hits = idx.search("anything")
        assert hits == []

    def test_clear(self) -> None:
        idx = DocumentationIndex(embedder=HashEmbeddingProvider())
        idx.ingest("Some content here.", source="s.txt", doc_format=DocFormat.TEXT)
        assert idx.chunk_count > 0

        idx.clear()
        assert idx.chunk_count == 0
        assert idx.indexed_sources == set()

    def test_source_tracking(self) -> None:
        idx = DocumentationIndex(embedder=HashEmbeddingProvider())
        idx.ingest("Content A", source="a.txt", doc_format=DocFormat.TEXT)
        idx.ingest("Content B", source="b.txt", doc_format=DocFormat.TEXT)
        assert idx.indexed_sources == {"a.txt", "b.txt"}

    def test_ingest_empty_returns_zero(self) -> None:
        idx = DocumentationIndex(embedder=HashEmbeddingProvider())
        count = idx.ingest("", source="empty.txt", doc_format=DocFormat.TEXT)
        assert count == 0
        assert idx.chunk_count == 0

    def test_search_with_section_filter(self) -> None:
        spec = json.dumps({
            "openapi": "3.0.0",
            "info": {"title": "Test", "version": "1.0.0"},
            "paths": {
                "/users": {
                    "get": {"summary": "List users", "responses": {"200": {"description": "OK"}}},
                    "post": {"summary": "Create user", "responses": {"201": {"description": "Created"}}},
                },
            },
        })
        idx = DocumentationIndex(embedder=HashEmbeddingProvider())
        idx.ingest(spec, source="api.json", doc_format=DocFormat.OPENAPI)

        hits = idx.search("users", section_filter="paths./users.get")
        assert len(hits) > 0
        for h in hits:
            assert h.chunk.section == "paths./users.get"
