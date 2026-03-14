"""Embedding providers for the RAG system.

Includes a zero-dependency hash-based provider for dev/testing and
an optional OpenAI provider for production use.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    @property
    def dimensions(self) -> int:
        """Dimensionality of the embedding vectors."""
        ...

    def embed(self, text: str) -> list[float]:
        """Embed a single text string into a vector."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts at once."""
        ...


class HashEmbeddingProvider:
    """Deterministic hash-based embeddings for dev/testing.

    Uses SHA-256 to produce consistent embeddings. Not semantically
    meaningful, but deterministic and zero-dependency.
    """

    def __init__(self, dimensions: int = 64) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, text: str) -> list[float]:
        """Hash text into a deterministic vector."""
        text_lower = text.lower().strip()
        digest = hashlib.sha256(text_lower.encode("utf-8")).digest()

        # Expand hash to fill desired dimensions
        raw: list[float] = []
        for i in range(self._dimensions):
            byte_idx = i % len(digest)
            raw.append((digest[byte_idx] + i * 37) % 256 / 255.0 * 2.0 - 1.0)

        # L2 normalize
        magnitude = math.sqrt(sum(x * x for x in raw))
        if magnitude > 0:
            raw = [x / magnitude for x in raw]
        return raw

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class OpenAIEmbeddingProvider:
    """OpenAI text-embedding-3-small provider.

    Requires the ``openai`` package (optional dependency).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
    ) -> None:
        try:
            import openai  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "OpenAI provider requires the 'openai' package. "
                "Install with: pip install mcp-api-wrapper[openai]"
            ) from exc

        self._client: object = openai.OpenAI(api_key=api_key)  # type: ignore[no-untyped-call] # openai is an optional untyped dep
        self._model = model
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        import openai  # type: ignore[import-untyped]

        client: openai.OpenAI = self._client  # type: ignore[assignment]  # openai is untyped optional dep
        response = client.embeddings.create(input=texts, model=self._model)  # type: ignore[reportUnknownMemberType]  # openai untyped
        return [item.embedding for item in response.data]  # type: ignore[reportUnknownMemberType, reportUnknownVariableType]  # openai untyped
