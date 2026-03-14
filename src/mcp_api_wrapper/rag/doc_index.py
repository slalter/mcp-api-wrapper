"""Documentation index: orchestrates chunking, embedding, and search."""

from __future__ import annotations

from mcp_api_wrapper.rag.chunking import DocumentChunker
from mcp_api_wrapper.rag.embeddings import EmbeddingProvider, HashEmbeddingProvider
from mcp_api_wrapper.rag.types import DocFormat, SearchHit
from mcp_api_wrapper.rag.vector_store import InMemoryVectorStore, VectorStore


class DocumentationIndex:
    """Orchestrates document ingestion and semantic search.

    Ties together chunker + embedder + vector store into
    a single high-level API for the MCP tool.
    """

    def __init__(
        self,
        embedder: EmbeddingProvider | None = None,
        store: VectorStore | None = None,
        max_chunk_size: int = 1500,
        chunk_overlap: int = 200,
    ) -> None:
        self._embedder: EmbeddingProvider = embedder or HashEmbeddingProvider()
        self._store: VectorStore = store or InMemoryVectorStore()
        self._chunker = DocumentChunker(
            max_chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._indexed_sources: set[str] = set()

    @property
    def chunk_count(self) -> int:
        """Total number of indexed chunks."""
        return self._store.count()

    @property
    def indexed_sources(self) -> set[str]:
        """Set of sources that have been indexed."""
        return set(self._indexed_sources)

    def ingest(
        self,
        content: str,
        source: str,
        doc_format: DocFormat,
    ) -> int:
        """Chunk, embed, and store a document.

        Returns the number of chunks created.
        """
        chunks = self._chunker.chunk(content, source, doc_format)
        if not chunks:
            return 0

        texts = [c.content for c in chunks]
        embeddings = self._embedder.embed_batch(texts)
        self._store.add_batch(chunks, embeddings)
        self._indexed_sources.add(source)
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        section_filter: str | None = None,
    ) -> list[SearchHit]:
        """Search indexed documentation.

        Returns ranked list of SearchHit objects.
        """
        if self._store.count() == 0:
            return []

        query_embedding = self._embedder.embed(query)
        return self._store.search(
            query_embedding,
            top_k=top_k,
            min_score=min_score,
            section_filter=section_filter,
        )

    def clear(self) -> None:
        """Remove all indexed documents."""
        self._store.clear()
        self._indexed_sources.clear()
