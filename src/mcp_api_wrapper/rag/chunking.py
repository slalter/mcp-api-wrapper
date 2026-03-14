"""Document chunking strategies for different formats."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, cast

from mcp_api_wrapper.rag.types import DocChunk, DocFormat


class DocumentChunker:
    """Splits documents into chunks based on format."""

    def __init__(
        self,
        max_chunk_size: int = 1500,
        chunk_overlap: int = 200,
    ) -> None:
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(
        self,
        content: str,
        source: str,
        doc_format: DocFormat,
    ) -> list[DocChunk]:
        """Chunk a document based on its format."""
        if not content.strip():
            return []

        if doc_format == DocFormat.OPENAPI:
            return self._chunk_openapi(content, source)
        elif doc_format == DocFormat.MARKDOWN:
            return self._chunk_markdown(content, source)
        elif doc_format == DocFormat.HTML:
            return self._chunk_html(content, source)
        else:
            return self._chunk_text(content, source, doc_format)

    def _chunk_openapi(self, content: str, source: str) -> list[DocChunk]:
        """Chunk OpenAPI spec: one chunk per path+method, plus info and schemas."""
        try:
            spec: dict[str, Any] = json.loads(content)
        except json.JSONDecodeError:
            return self._chunk_text(content, source, DocFormat.OPENAPI)

        chunks: list[DocChunk] = []

        # Info/description chunk
        info: dict[str, Any] = spec.get("info", {})
        if info:
            info_text = f"API: {info.get('title', 'Unknown')}\n"
            info_text += f"Version: {info.get('version', 'Unknown')}\n"
            if info.get("description"):
                info_text += f"Description: {info['description']}\n"
            chunks.append(
                DocChunk(
                    chunk_id=_make_id(),
                    content=info_text,
                    source=source,
                    section="info",
                    doc_format=DocFormat.OPENAPI,
                )
            )

        # One chunk per path+method
        paths: dict[str, Any] = spec.get("paths", {})
        for path_str, methods_obj in paths.items():
            if not isinstance(methods_obj, dict):
                continue
            methods_dict = cast(dict[str, Any], methods_obj)
            for method_str, details_obj in methods_dict.items():
                if str(method_str).startswith("x-") or not isinstance(details_obj, dict):
                    continue
                details = cast(dict[str, Any], details_obj)
                op_text = f"{str(method_str).upper()} {path_str}\n"
                if details.get("summary"):
                    op_text += f"Summary: {details['summary']}\n"
                if details.get("description"):
                    op_text += f"Description: {details['description']}\n"
                if details.get("parameters"):
                    op_text += "Parameters:\n"
                    params: list[Any] = details["parameters"]
                    for param in params:
                        if isinstance(param, dict):
                            p = cast(dict[str, Any], param)
                            op_text += f"  - {p.get('name', '?')} ({p.get('in', '?')}): {p.get('description', '')}\n"
                if details.get("requestBody"):
                    op_text += "Request Body: required\n"
                if details.get("responses"):
                    op_text += "Responses:\n"
                    responses = cast(dict[str, Any], details["responses"])
                    for code, resp in responses.items():
                        resp_dict = cast(dict[str, Any], resp) if isinstance(resp, dict) else {}
                        desc: str = str(resp_dict.get("description", "")) if resp_dict else ""
                        op_text += f"  {code}: {desc}\n"

                section = f"paths.{path_str}.{method_str}"
                chunk_content = op_text.strip()
                if len(chunk_content) > self.max_chunk_size:
                    chunks.extend(
                        self._fixed_size_split(chunk_content, source, section, DocFormat.OPENAPI)
                    )
                else:
                    chunks.append(
                        DocChunk(
                            chunk_id=_make_id(),
                            content=chunk_content,
                            source=source,
                            section=section,
                            doc_format=DocFormat.OPENAPI,
                        )
                    )

        # Component schemas
        components: dict[str, Any] = spec.get("components", {})
        schemas: dict[str, Any] = components.get("schemas", {})
        for schema_name, schema_val in schemas.items():
            if not isinstance(schema_val, dict):
                continue
            schema_text = f"Schema: {schema_name}\n{json.dumps(schema_val, indent=2)}"
            section = f"schemas.{schema_name}"
            if len(schema_text) > self.max_chunk_size:
                chunks.extend(
                    self._fixed_size_split(schema_text, source, section, DocFormat.OPENAPI)
                )
            else:
                chunks.append(
                    DocChunk(
                        chunk_id=_make_id(),
                        content=schema_text,
                        source=source,
                        section=section,
                        doc_format=DocFormat.OPENAPI,
                    )
                )

        return chunks

    def _chunk_markdown(self, content: str, source: str) -> list[DocChunk]:
        """Chunk markdown by heading hierarchy."""
        # Split by headings (# through ####)
        heading_pattern = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
        matches = list(heading_pattern.finditer(content))

        if not matches:
            return self._chunk_text(content, source, DocFormat.MARKDOWN)

        chunks: list[DocChunk] = []

        # Content before first heading
        if matches[0].start() > 0:
            pre = content[: matches[0].start()].strip()
            if pre:
                chunks.append(
                    DocChunk(
                        chunk_id=_make_id(),
                        content=pre,
                        source=source,
                        section="preamble",
                        doc_format=DocFormat.MARKDOWN,
                    )
                )

        # Each heading section
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            section_content = content[start:end].strip()
            section_name = match.group(2).strip()

            if len(section_content) > self.max_chunk_size:
                chunks.extend(
                    self._fixed_size_split(section_content, source, section_name, DocFormat.MARKDOWN)
                )
            else:
                chunks.append(
                    DocChunk(
                        chunk_id=_make_id(),
                        content=section_content,
                        source=source,
                        section=section_name,
                        doc_format=DocFormat.MARKDOWN,
                    )
                )

        return chunks

    def _chunk_html(self, content: str, source: str) -> list[DocChunk]:
        """Strip HTML tags and delegate to markdown chunker."""
        # Simple tag stripping (no dependency on beautifulsoup)
        text = re.sub(r"<[^>]+>", "", content)
        text = re.sub(r"\s+", " ", text).strip()
        return self._chunk_markdown(text, source)

    def _chunk_text(
        self,
        content: str,
        source: str,
        doc_format: DocFormat,
    ) -> list[DocChunk]:
        """Fixed-size chunking with overlap."""
        return self._fixed_size_split(content, source, "text", doc_format)

    def _fixed_size_split(
        self,
        content: str,
        source: str,
        section: str,
        doc_format: DocFormat,
    ) -> list[DocChunk]:
        """Split text into fixed-size chunks with overlap."""
        chunks: list[DocChunk] = []
        start = 0
        while start < len(content):
            end = start + self.max_chunk_size
            chunk_text = content[start:end].strip()
            if chunk_text:
                chunks.append(
                    DocChunk(
                        chunk_id=_make_id(),
                        content=chunk_text,
                        source=source,
                        section=section,
                        doc_format=doc_format,
                    )
                )
            start += self.max_chunk_size - self.chunk_overlap
        return chunks


def _make_id() -> str:
    """Generate a short unique chunk ID."""
    return uuid.uuid4().hex[:12]
