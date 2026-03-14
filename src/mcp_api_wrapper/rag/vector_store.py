"""Vector store implementations for similarity search."""

from __future__ import annotations

import math
from typing import Protocol

from mcp_api_wrapper.rag.types import DocChunk, SearchHit


class VectorStore(Protocol):
    """Protocol for vector stores."""

    def add(self, chunk: DocChunk, embedding: list[float]) -> None:
        """Add a chunk with its embedding."""
        ...

    def add_batch(self, chunks: list[DocChunk], embeddings: list[list[float]]) -> None:
        """Add multiple chunks with their embeddings."""
        ...

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        min_score: float = 0.0,
        section_filter: str | None = None,
    ) -> list[SearchHit]:
        """Search for similar chunks."""
        ...

    def clear(self) -> None:
        """Remove all stored chunks."""
        ...

    def count(self) -> int:
        """Number of stored chunks."""
        ...


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        raise ValueError(f"Vector dimensions differ: {len(a)} vs {len(b)}")

    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))

    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class InMemoryVectorStore:
    """Brute-force in-memory vector store using cosine similarity.

    Zero external dependencies. Suitable for small-to-medium document sets.
    """

    def __init__(self) -> None:
        self._chunks: list[DocChunk] = []
        self._embeddings: list[list[float]] = []

    def add(self, chunk: DocChunk, embedding: list[float]) -> None:
        self._chunks.append(chunk)
        self._embeddings.append(embedding)

    def add_batch(self, chunks: list[DocChunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        self._chunks.extend(chunks)
        self._embeddings.extend(embeddings)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        min_score: float = 0.0,
        section_filter: str | None = None,
    ) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for chunk, emb in zip(self._chunks, self._embeddings):
            if section_filter and chunk.section != section_filter:
                continue
            score = _cosine_similarity(query_embedding, emb)
            if score >= min_score:
                hits.append(SearchHit(chunk=chunk, score=score))

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def clear(self) -> None:
        self._chunks.clear()
        self._embeddings.clear()

    def count(self) -> int:
        return len(self._chunks)
