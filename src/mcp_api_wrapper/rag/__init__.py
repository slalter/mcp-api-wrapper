"""RAG (Retrieval-Augmented Generation) module for documentation search."""

from mcp_api_wrapper.rag.chunking import DocumentChunker
from mcp_api_wrapper.rag.doc_index import DocumentationIndex
from mcp_api_wrapper.rag.embeddings import (
    EmbeddingProvider,
    HashEmbeddingProvider,
    OpenAIEmbeddingProvider,
)
from mcp_api_wrapper.rag.types import DocChunk, DocFormat, SearchHit
from mcp_api_wrapper.rag.vector_store import InMemoryVectorStore, VectorStore

__all__ = [
    "DocChunk",
    "DocFormat",
    "DocumentChunker",
    "DocumentationIndex",
    "EmbeddingProvider",
    "HashEmbeddingProvider",
    "InMemoryVectorStore",
    "OpenAIEmbeddingProvider",
    "SearchHit",
    "VectorStore",
]
