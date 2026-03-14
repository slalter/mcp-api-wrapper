"""Tests for document chunking strategies."""

from __future__ import annotations

import json

import pytest

from mcp_api_wrapper.rag.chunking import DocumentChunker
from mcp_api_wrapper.rag.types import DocFormat


@pytest.fixture
def chunker() -> DocumentChunker:
    return DocumentChunker(max_chunk_size=500, chunk_overlap=50)


@pytest.fixture
def sample_openapi() -> str:
    return json.dumps({
        "openapi": "3.0.0",
        "info": {
            "title": "Test API",
            "version": "1.0.0",
            "description": "A test API for chunking",
        },
        "paths": {
            "/users": {
                "get": {
                    "summary": "List users",
                    "description": "Returns all users",
                    "parameters": [
                        {"name": "limit", "in": "query", "description": "Max results"},
                    ],
                    "responses": {"200": {"description": "Success"}},
                },
                "post": {
                    "summary": "Create user",
                    "requestBody": {"required": True},
                    "responses": {
                        "201": {"description": "Created"},
                        "400": {"description": "Bad request"},
                    },
                },
            },
            "/users/{id}": {
                "get": {
                    "summary": "Get user by ID",
                    "responses": {"200": {"description": "Success"}},
                },
            },
        },
        "components": {
            "schemas": {
                "User": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "name": {"type": "string"},
                    },
                },
            },
        },
    })


class TestOpenAPIChunking:
    def test_creates_info_chunk(self, chunker: DocumentChunker, sample_openapi: str) -> None:
        chunks = chunker.chunk(sample_openapi, "test.json", DocFormat.OPENAPI)
        info_chunks = [c for c in chunks if c.section == "info"]
        assert len(info_chunks) == 1
        assert "Test API" in info_chunks[0].content
        assert "1.0.0" in info_chunks[0].content

    def test_creates_per_path_method_chunks(
        self, chunker: DocumentChunker, sample_openapi: str
    ) -> None:
        chunks = chunker.chunk(sample_openapi, "test.json", DocFormat.OPENAPI)
        path_chunks = [c for c in chunks if c.section.startswith("paths.")]
        # 3 operations: GET /users, POST /users, GET /users/{id}
        assert len(path_chunks) == 3

    def test_includes_parameters_in_chunk(
        self, chunker: DocumentChunker, sample_openapi: str
    ) -> None:
        chunks = chunker.chunk(sample_openapi, "test.json", DocFormat.OPENAPI)
        get_users = [c for c in chunks if c.section == "paths./users.get"]
        assert len(get_users) == 1
        assert "limit" in get_users[0].content
        assert "query" in get_users[0].content

    def test_includes_request_body(
        self, chunker: DocumentChunker, sample_openapi: str
    ) -> None:
        chunks = chunker.chunk(sample_openapi, "test.json", DocFormat.OPENAPI)
        post_users = [c for c in chunks if c.section == "paths./users.post"]
        assert len(post_users) == 1
        assert "Request Body" in post_users[0].content

    def test_creates_schema_chunk(
        self, chunker: DocumentChunker, sample_openapi: str
    ) -> None:
        chunks = chunker.chunk(sample_openapi, "test.json", DocFormat.OPENAPI)
        schema_chunks = [c for c in chunks if c.section.startswith("schemas.")]
        assert len(schema_chunks) == 1
        assert "User" in schema_chunks[0].content

    def test_all_chunks_have_correct_format(
        self, chunker: DocumentChunker, sample_openapi: str
    ) -> None:
        chunks = chunker.chunk(sample_openapi, "test.json", DocFormat.OPENAPI)
        for chunk in chunks:
            assert chunk.doc_format == DocFormat.OPENAPI
            assert chunk.source == "test.json"

    def test_invalid_json_falls_back_to_text(self, chunker: DocumentChunker) -> None:
        chunks = chunker.chunk("not valid json", "test.json", DocFormat.OPENAPI)
        assert len(chunks) >= 1
        assert chunks[0].content == "not valid json"


class TestMarkdownChunking:
    def test_splits_by_headings(self, chunker: DocumentChunker) -> None:
        md = "# Introduction\nSome intro text.\n\n# API Reference\nAPI details here.\n"
        chunks = chunker.chunk(md, "docs.md", DocFormat.MARKDOWN)
        assert len(chunks) == 2
        sections = {c.section for c in chunks}
        assert "Introduction" in sections
        assert "API Reference" in sections

    def test_preserves_heading_hierarchy(self, chunker: DocumentChunker) -> None:
        md = "# Main\nText\n## Sub\nMore text\n### Deep\nDeep text\n"
        chunks = chunker.chunk(md, "docs.md", DocFormat.MARKDOWN)
        assert len(chunks) == 3

    def test_preamble_before_first_heading(self, chunker: DocumentChunker) -> None:
        md = "Some preamble text.\n\n# First Heading\nContent.\n"
        chunks = chunker.chunk(md, "docs.md", DocFormat.MARKDOWN)
        preamble = [c for c in chunks if c.section == "preamble"]
        assert len(preamble) == 1
        assert "preamble" in preamble[0].content.lower() or "Some preamble" in preamble[0].content

    def test_no_headings_falls_back_to_text(self, chunker: DocumentChunker) -> None:
        md = "Just plain text without any headings at all."
        chunks = chunker.chunk(md, "docs.md", DocFormat.MARKDOWN)
        assert len(chunks) >= 1


class TestTextChunking:
    def test_fixed_size_split(self) -> None:
        chunker = DocumentChunker(max_chunk_size=100, chunk_overlap=20)
        text = "a" * 250
        chunks = chunker.chunk(text, "file.txt", DocFormat.TEXT)
        # With 100 char chunks and 20 overlap, step = 80
        # chunks at: 0-100, 80-180, 160-250, 240-250 → 4 chunks
        assert len(chunks) >= 3

    def test_overlap_produces_shared_content(self) -> None:
        chunker = DocumentChunker(max_chunk_size=100, chunk_overlap=20)
        text = "".join(str(i % 10) for i in range(200))
        chunks = chunker.chunk(text, "file.txt", DocFormat.TEXT)
        # Verify overlap: end of chunk 0 should overlap with start of chunk 1
        assert len(chunks) >= 2
        tail_0 = chunks[0].content[-20:]
        head_1 = chunks[1].content[:20]
        assert tail_0 == head_1


class TestEdgeCases:
    def test_empty_document(self, chunker: DocumentChunker) -> None:
        chunks = chunker.chunk("", "empty.txt", DocFormat.TEXT)
        assert chunks == []

    def test_whitespace_only(self, chunker: DocumentChunker) -> None:
        chunks = chunker.chunk("   \n\n  ", "ws.txt", DocFormat.TEXT)
        assert chunks == []

    def test_oversized_openapi_section_splits(self) -> None:
        """A single operation with a very long description should be split."""
        chunker = DocumentChunker(max_chunk_size=100, chunk_overlap=20)
        spec = json.dumps({
            "openapi": "3.0.0",
            "info": {"title": "T", "version": "1.0.0"},
            "paths": {
                "/big": {
                    "get": {
                        "summary": "Big endpoint",
                        "description": "x" * 300,
                        "responses": {"200": {"description": "OK"}},
                    },
                },
            },
        })
        chunks = chunker.chunk(spec, "big.json", DocFormat.OPENAPI)
        big_chunks = [c for c in chunks if c.section.startswith("paths./big")]
        # Should have been split into multiple chunks
        assert len(big_chunks) > 1
