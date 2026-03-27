"""Chunk creation helpers for canonical ingestion workflows."""

from __future__ import annotations

from botadvisor.app.core.entity.chunk import Chunk
from botadvisor.app.core.entity.document import Document
from botadvisor.app.core.entity.document_metadata import DocumentMetadata


def create_chunks(text: str, *, document: Document, chunk_size: int = 1000) -> list[Chunk]:
    """Split text into chunk entities while preserving document metadata."""
    chunks: list[Chunk] = []
    for index, start in enumerate(range(0, len(text), chunk_size)):
        chunk_content = text[start : start + chunk_size]
        chunks.append(
            Chunk(
                chunk_id=f"{document.id}_chunk_{index}",
                content=chunk_content.strip(),
                metadata=DocumentMetadata(
                    doc_id=document.id,
                    source_id=document.source_id,
                    platform=document.platform,
                    source_url=document.source_url,
                    page=None,
                    section=f"chunk_{index}",
                ),
            )
        )
    return chunks
