"""Semantic chunking use case integrating the LlamaIndex-based chunker (Task 2.1.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence
from uuid import UUID

from app.config.ml_models import get_chunking_config
from app.domain.entities.document import Document
from app.domain.entities.chunk import Chunk
from app.domain.repositories.document_repository import DocumentRepository
from app.domain.value_objects.chunk_metadata import ChunkMetadata
from app.infrastructure.external_services.ml_services.chunking_services.llama_index_chunker import (
    LlamaIndexChunker,
    LlamaIndexChunkerConfig,
    SemanticChunk,
)


def _document_metadata_dict(document: Document) -> Dict[str, Any]:
    metadata = document.metadata
    return {
        "file_name": metadata.file_name,
        "file_path": metadata.file_path,
        "doc_type": metadata.format.value,
        "language": metadata.language,
        "title": metadata.title,
        "author": metadata.author,
        "keywords": metadata.keywords,
        "custom_fields": metadata.custom_fields,
    }


def _default_chunk_metadata(document: Document) -> Dict[str, Any]:
    return {
        "document_id": str(document.id),
        "source_id": str(document.source_id),
        "doc_type": document.metadata.format.value,
        "file_name": document.metadata.file_name,
    }


@dataclass
class SemanticChunkingRequest:
    """Semantic chunking request payload."""

    document_ids: Optional[Sequence[UUID]] = None
    include_failed: bool = False
    replace_existing_chunks: bool = True
    dry_run: bool = False
    chunker_config: Optional[Dict[str, Any]] = None
    metadata_overrides: Optional[Dict[str, Any]] = None


@dataclass
class DocumentChunkSummary:
    document_id: UUID
    chunk_count: int
    dry_run: bool
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "document_id": str(self.document_id),
            "chunk_count": self.chunk_count,
            "dry_run": self.dry_run,
            "warnings": self.warnings,
        }


@dataclass
class SemanticChunkingResponse:
    success: bool
    results: List[DocumentChunkSummary]
    total_chunks: int
    dry_run: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "total_chunks": self.total_chunks,
            "dry_run": self.dry_run,
            "results": [item.as_dict() for item in self.results],
        }


class SemanticChunkingUseCase:
    """Orchestrate semantic chunking for processed documents (Task 2.1.1).

    Examples:
        >>> use_case = SemanticChunkingUseCase(document_repository)
        >>> request = SemanticChunkingRequest(document_ids=[uuid4()])
        >>> response = await use_case.execute(request)
        >>> response.success
        True
    """

    def __init__(
        self,
        document_repository: DocumentRepository,
        chunker: Optional[LlamaIndexChunker] = None,
    ) -> None:
        self._repository = document_repository
        self._chunker = chunker

    async def execute(self, request: SemanticChunkingRequest) -> SemanticChunkingResponse:
        documents = await self._load_documents(request)
        if not documents:
            return SemanticChunkingResponse(
                success=True,
                results=[],
                total_chunks=0,
                dry_run=request.dry_run,
            )

        config_payload = request.chunker_config or get_chunking_config()
        chunker = self._chunker or LlamaIndexChunker(
            config=LlamaIndexChunkerConfig.from_dict(config_payload),
            require_llama_index=False,
        )

        results: List[DocumentChunkSummary] = []
        total_chunks = 0

        for document in documents:
            chunks = self._chunk_document(chunker, document, request)
            chunk_count = len(chunks)
            total_chunks += chunk_count
            result = DocumentChunkSummary(
                document_id=document.id,
                chunk_count=chunk_count,
                dry_run=request.dry_run,
            )
            results.append(result)

            if not request.dry_run:
                serialized_chunks = [chunk.as_dict() for chunk in chunks]
                if request.replace_existing_chunks or not document.chunks:
                    document.set_chunks(serialized_chunks)
                else:
                    merged = document.chunks + serialized_chunks
                    document.set_chunks(merged)
                await self._repository.save(document)

        return SemanticChunkingResponse(
            success=True,
            results=results,
            total_chunks=total_chunks,
            dry_run=request.dry_run,
        )

    async def _load_documents(self, request: SemanticChunkingRequest) -> List[Document]:
        if request.document_ids:
            documents: List[Document] = []
            for doc_id in request.document_ids:
                doc = await self._repository.find_by_id(doc_id)
                if doc is not None:
                    documents.append(doc)
            return documents

        if request.include_failed:
            return await self._repository.find_failed_documents()

        return await self._repository.find_processed_documents()

    def _chunk_document(
        self,
        chunker: LlamaIndexChunker,
        document: Document,
        request: SemanticChunkingRequest,
    ) -> List[Chunk]:
        text_candidates: Iterable[Optional[str]] = (
            document.extracted_text,
            document.content,
        )
        text = next((value for value in text_candidates if value), "")

        metadata_payload = _default_chunk_metadata(document)
        if request.metadata_overrides:
            metadata_payload.update(request.metadata_overrides)

        document_metadata = _document_metadata_dict(document)

        semantic_chunks = chunker.chunk(
            text,
            metadata=metadata_payload,
            document_metadata=document_metadata,
        )

        return [self._semantic_to_chunk(document, chunk) for chunk in semantic_chunks]

    def _semantic_to_chunk(self, document: Document, semantic_chunk: SemanticChunk) -> Chunk:
        metadata_copy = dict(semantic_chunk.metadata)
        token_count = metadata_copy.pop("token_count", None)
        section = metadata_copy.pop("section", None)
        heading_path = metadata_copy.pop("heading_path", None)
        page_number = metadata_copy.pop("page_number", None)
        source_type = metadata_copy.pop("source_type", None)
        similarity_score = metadata_copy.pop("similarity_score", None)

        chunk_metadata = ChunkMetadata(
            sequence=semantic_chunk.sequence,
            start=semantic_chunk.start,
            end=semantic_chunk.end,
            token_count=token_count,
            section=section,
            heading_path=heading_path if heading_path else None,
            page_number=page_number,
            source_type=source_type,
            similarity_score=similarity_score,
            extra=metadata_copy,
        )

        return Chunk(
            chunk_id=semantic_chunk.chunk_id,
            document_id=document.id,
            text=semantic_chunk.text,
            metadata=chunk_metadata,
        )
