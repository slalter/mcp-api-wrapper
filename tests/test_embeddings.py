"""Tests for embedding providers."""

from __future__ import annotations

import math

from mcp_api_wrapper.rag.embeddings import HashEmbeddingProvider


class TestHashEmbeddingProvider:
    def test_deterministic(self) -> None:
        """Same text always produces same embedding."""
        provider = HashEmbeddingProvider(dimensions=64)
        v1 = provider.embed("hello world")
        v2 = provider.embed("hello world")
        assert v1 == v2

    def test_case_insensitive(self) -> None:
        """Embeddings are case-insensitive."""
        provider = HashEmbeddingProvider(dimensions=64)
        v1 = provider.embed("Hello World")
        v2 = provider.embed("hello world")
        assert v1 == v2

    def test_different_texts_differ(self) -> None:
        """Different texts produce different embeddings."""
        provider = HashEmbeddingProvider(dimensions=64)
        v1 = provider.embed("hello world")
        v2 = provider.embed("goodbye world")
        assert v1 != v2

    def test_correct_dimensions(self) -> None:
        provider = HashEmbeddingProvider(dimensions=128)
        v = provider.embed("test")
        assert len(v) == 128

    def test_normalized(self) -> None:
        """Embeddings should be L2-normalized (magnitude ~1.0)."""
        provider = HashEmbeddingProvider(dimensions=64)
        v = provider.embed("test embedding")
        magnitude = math.sqrt(sum(x * x for x in v))
        assert abs(magnitude - 1.0) < 1e-6

    def test_batch(self) -> None:
        provider = HashEmbeddingProvider(dimensions=64)
        texts = ["alpha", "beta", "gamma"]
        batch_result = provider.embed_batch(texts)
        assert len(batch_result) == 3
        # Batch should match individual calls
        for text, batch_vec in zip(texts, batch_result):
            assert provider.embed(text) == batch_vec

    def test_default_dimensions(self) -> None:
        provider = HashEmbeddingProvider()
        v = provider.embed("test")
        assert len(v) == 64
        assert provider.dimensions == 64
