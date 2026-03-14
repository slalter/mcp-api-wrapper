"""Core types for the RAG documentation search system."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DocFormat(str, Enum):
    """Supported documentation formats."""

    OPENAPI = "openapi"
    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"


@dataclass(frozen=True)
class DocChunk:
    """A single chunk of documentation content."""

    chunk_id: str
    content: str
    source: str
    section: str
    doc_format: DocFormat
    metadata: dict[str, str] = field(default_factory=lambda: dict[str, str]())


@dataclass
class SearchHit:
    """A search result with relevance score."""

    chunk: DocChunk
    score: float
