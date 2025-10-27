"""Recursive chunking use case for hierarchical documents (Task 2.1.3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID

from app.config.ml_models import get_chunking_config
from app.domain.entities.chunk import Chunk
from app.domain.entities.document import Document
from app.domain.repositories.document_repository import DocumentRepository
from app.domain.value_objects.chunk_metadata import ChunkMetadata
from app.infrastructure.external_services.ml_services.chunking_services.llama_hierarchical_chunker import (
    HierarchicalChunk,
    HierarchicalChunkerConfig,
    LlamaHierarchicalChunker,
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
class RecursiveChunkingRequest:
    """Recursive chunking request payload for hierarchical documents."""

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
    hierarchy_stats: Dict[str, int]  # Count by depth level
    dry_run: bool
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "document_id": str(self.document_id),
            "chunk_count": self.chunk_count,
            "hierarchy_stats": self.hierarchy_stats,
            "dry_run": self.dry_run,
            "warnings": self.warnings,
        }


@dataclass
class RecursiveChunkingResponse:
    success: bool
    results: List[DocumentChunkSummary]
    total_chunks: int
    total_hierarchy_stats: Dict[str, int]  # Aggregated count by depth level
    dry_run: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "total_chunks": self.total_chunks,
            "total_hierarchy_stats": self.total_hierarchy_stats,
            "dry_run": self.dry_run,
            "results": [item.as_dict() for item in self.results],
        }


class RecursiveChunkingUseCase:
    """Orchestrate recursive chunking for hierarchical documents (Task 2.1.3).

    Uses llama_index HierarchicalNodeParser to create multi-level chunks
    preserving policy > section > paragraph hierarchy.

    Examples:
        >>> use_case = RecursiveChunkingUseCase(document_repository)
        >>> request = RecursiveChunkingRequest(document_ids=[uuid4()])
        >>> response = await use_case.execute(request)
        >>> response.success
        True
        >>> "depth_0" in response.total_hierarchy_stats
        True
    """

    def __init__(
        self,
        document_repository: DocumentRepository,
        chunker: Optional[LlamaHierarchicalChunker] = None,
    ) -> None:
        self._repository = document_repository
        self._chunker = chunker

    async def execute(self, request: RecursiveChunkingRequest) -> RecursiveChunkingResponse:
        documents = await self._load_documents(request)
        if not documents:
            return RecursiveChunkingResponse(
                success=True,
                results=[],
                total_chunks=0,
                total_hierarchy_stats={},
                dry_run=request.dry_run,
            )

        config_payload = request.chunker_config or get_chunking_config("hierarchical")

        # Convert chunk_sizes from config if present
        if "chunk_sizes" in config_payload and isinstance(config_payload["chunk_sizes"], str):
            # Handle string representation like "[2048, 512, 128]"
            import ast
            try:
                config_payload["chunk_sizes"] = ast.literal_eval(config_payload["chunk_sizes"])
            except (ValueError, SyntaxError):
                pass

        chunker = self._chunker or LlamaHierarchicalChunker(
            config=HierarchicalChunkerConfig.from_dict(config_payload),
            require_llama_index=False,
        )

        results: List[DocumentChunkSummary] = []
        total_chunks = 0
        total_hierarchy_stats: Dict[str, int] = {}

        for document in documents:
            chunks = self._chunk_document(chunker, document, request)
            chunk_count = len(chunks)
            total_chunks += chunk_count

            # Calculate hierarchy statistics
            hierarchy_stats = self._calculate_hierarchy_stats(chunks)
            for depth_key, count in hierarchy_stats.items():
                total_hierarchy_stats[depth_key] = total_hierarchy_stats.get(depth_key, 0) + count

            result = DocumentChunkSummary(
                document_id=document.id,
                chunk_count=chunk_count,
                hierarchy_stats=hierarchy_stats,
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

        return RecursiveChunkingResponse(
            success=True,
            results=results,
            total_chunks=total_chunks,
            total_hierarchy_stats=total_hierarchy_stats,
            dry_run=request.dry_run,
        )

    async def _load_documents(self, request: RecursiveChunkingRequest) -> List[Document]:
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
        chunker: LlamaHierarchicalChunker,
        document: Document,
        request: RecursiveChunkingRequest,
    ) -> List[Chunk]:
        # Extract text from document
        text_candidates = (
            document.extracted_text,
            document.content,
        )
        text = next((value for value in text_candidates if value), "")

        # Prepare metadata
        metadata_payload = _default_chunk_metadata(document)
        if request.metadata_overrides:
            metadata_payload.update(request.metadata_overrides)

        document_metadata = _document_metadata_dict(document)

        # Perform hierarchical chunking
        hierarchical_chunks = chunker.chunk(
            text,
            metadata=metadata_payload,
            document_metadata=document_metadata,
        )

        # Convert to domain Chunk entities
        return [self._hierarchical_to_chunk(document, hchunk) for hchunk in hierarchical_chunks]

    def _hierarchical_to_chunk(self, document: Document, hierarchical_chunk: HierarchicalChunk) -> Chunk:
        """Convert HierarchicalChunk to domain Chunk entity."""
        metadata_copy = dict(hierarchical_chunk.metadata)

        # Extract standard metadata fields
        token_count = metadata_copy.pop("token_count", None)
        section = metadata_copy.pop("section", None)
        heading_path = metadata_copy.pop("heading_path", None)
        page_number = metadata_copy.pop("page_number", None)
        source_type = metadata_copy.pop("source_type", None)
        similarity_score = metadata_copy.pop("similarity_score", None)

        # Add hierarchical metadata to extra
        metadata_copy["depth_level"] = hierarchical_chunk.depth_level
        if hierarchical_chunk.parent_chunk_id:
            metadata_copy["parent_id"] = hierarchical_chunk.parent_chunk_id

        chunk_metadata = ChunkMetadata(
            sequence=hierarchical_chunk.sequence,
            start=hierarchical_chunk.start,
            end=hierarchical_chunk.end,
            token_count=token_count,
            section=section,
            heading_path=heading_path if heading_path else None,
            page_number=page_number,
            source_type=source_type,
            similarity_score=similarity_score,
            extra=metadata_copy,
        )

        return Chunk(
            chunk_id=hierarchical_chunk.chunk_id,
            document_id=document.id,
            text=hierarchical_chunk.text,
            metadata=chunk_metadata,
        )

    def _calculate_hierarchy_stats(self, chunks: List[Chunk]) -> Dict[str, int]:
        """Calculate count of chunks at each depth level."""
        stats: Dict[str, int] = {}
        for chunk in chunks:
            depth_level = chunk.metadata.extra.get("depth_level", 0)
            depth_key = f"depth_{depth_level}"
            stats[depth_key] = stats.get(depth_key, 0) + 1
        return stats
