"""Tests for vector store implementations."""

from __future__ import annotations

import math

import pytest

from mcp_api_wrapper.rag.types import DocChunk, DocFormat, SearchHit
from mcp_api_wrapper.rag.vector_store import InMemoryVectorStore, _cosine_similarity


def _make_chunk(content: str, section: str = "test") -> DocChunk:
    return DocChunk(
        chunk_id="c1",
        content=content,
        source="test.txt",
        section=section,
        doc_format=DocFormat.TEXT,
    )


class TestCosineSimlarity:
    def test_identical_vectors(self) -> None:
        v = [1.0, 0.0, 0.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(_cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_zero_vector(self) -> None:
        a = [0.0, 0.0]
        b = [1.0, 0.0]
        assert _cosine_similarity(a, b) == 0.0

    def test_dimension_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="dimensions differ"):
            _cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


class TestInMemoryVectorStore:
    def test_add_and_search(self) -> None:
        store = InMemoryVectorStore()
        chunk = _make_chunk("hello")
        emb = [1.0, 0.0, 0.0]
        store.add(chunk, emb)

        results = store.search([1.0, 0.0, 0.0])
        assert len(results) == 1
        assert results[0].chunk == chunk
        assert abs(results[0].score - 1.0) < 1e-6

    def test_ranking_order(self) -> None:
        store = InMemoryVectorStore()
        c1 = _make_chunk("close match")
        c2 = _make_chunk("far match")

        # c1 is closer to query than c2
        store.add(c1, [0.9, 0.1, 0.0])
        store.add(c2, [0.1, 0.9, 0.0])

        results = store.search([1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0].chunk == c1
        assert results[0].score > results[1].score

    def test_top_k_limit(self) -> None:
        store = InMemoryVectorStore()
        for i in range(10):
            store.add(_make_chunk(f"chunk {i}"), [float(i) / 10, 0.5, 0.0])

        results = store.search([1.0, 0.0, 0.0], top_k=3)
        assert len(results) == 3

    def test_min_score_filter(self) -> None:
        store = InMemoryVectorStore()
        store.add(_make_chunk("high"), [1.0, 0.0, 0.0])
        store.add(_make_chunk("low"), [0.0, 1.0, 0.0])

        results = store.search([1.0, 0.0, 0.0], min_score=0.9)
        assert len(results) == 1
        assert results[0].chunk.content == "high"

    def test_section_filter(self) -> None:
        store = InMemoryVectorStore()
        store.add(_make_chunk("chunk A", section="alpha"), [1.0, 0.0, 0.0])
        store.add(_make_chunk("chunk B", section="beta"), [0.9, 0.1, 0.0])

        results = store.search([1.0, 0.0, 0.0], section_filter="beta")
        assert len(results) == 1
        assert results[0].chunk.section == "beta"

    def test_clear(self) -> None:
        store = InMemoryVectorStore()
        store.add(_make_chunk("x"), [1.0, 0.0])
        assert store.count() == 1
        store.clear()
        assert store.count() == 0
        assert store.search([1.0, 0.0]) == []

    def test_count(self) -> None:
        store = InMemoryVectorStore()
        assert store.count() == 0
        store.add(_make_chunk("a"), [1.0])
        store.add(_make_chunk("b"), [0.5])
        assert store.count() == 2

    def test_add_batch(self) -> None:
        store = InMemoryVectorStore()
        chunks = [_make_chunk("a"), _make_chunk("b"), _make_chunk("c")]
        embeddings = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        store.add_batch(chunks, embeddings)
        assert store.count() == 3

    def test_add_batch_mismatch_raises(self) -> None:
        store = InMemoryVectorStore()
        with pytest.raises(ValueError, match="same length"):
            store.add_batch([_make_chunk("a")], [[1.0], [2.0]])
