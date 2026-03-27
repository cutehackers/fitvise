"""Recursive chunking use case for hierarchical documents (Task 2.1.3).

Thread Safety Design:
- CPU-bound chunking: Dedicated ThreadPoolExecutor (via asyncio.to_thread)
- Repository saves: Serialized with asyncio.Lock (async safety)
- Overall concurrency: Limited by batch processing (max 4 concurrent)
- Cancellation: CancelledError propagated for clean shutdown
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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

logger = logging.getLogger(__name__)


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
    preserving policy > section > paragraph hierarchy. Processes documents
    in concurrent batches for improved performance.

    Thread Safety:
        - CPU-bound chunking runs in dedicated ThreadPoolExecutor
        - Repository saves serialized with asyncio.Lock
        - Batch processing limits overall concurrency (max 4 concurrent)
        - CancelledError propagated for clean cancellation

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
        max_concurrent_chunks: int = 4,
        thread_pool_size: int = 4,
    ) -> None:
        """Initialize recursive chunking use case.

        Args:
            document_repository: Repository for document persistence
            chunker: Optional pre-configured chunker. If not provided, created from config.
            max_concurrent_chunks: Max concurrent document chunks (default: 4).
                                  Controls parallelism to avoid memory overflow.
            thread_pool_size: Max threads in dedicated executor for CPU-bound work (default: 4).

        Thread Safety:
            - repository.save() calls serialized with asyncio.Lock
            - CPU-bound chunking runs in dedicated ThreadPoolExecutor
            - Cancellation (CancelledError) is propagated for clean shutdown
        """
        self._repository = document_repository
        self._chunker = chunker
        self._max_concurrent_chunks = max_concurrent_chunks

        # Thread safety mechanisms
        self._save_lock = asyncio.Lock()  # Serializes repository saves (created lazily)
        self._executor = ThreadPoolExecutor(
            max_workers=thread_pool_size, thread_name_prefix="recursive-worker"
        )  # Dedicated pool for CPU-bound work

    async def execute(self, request: RecursiveChunkingRequest) -> RecursiveChunkingResponse:
        """Execute hierarchical chunking with parallel document processing.

        Documents are chunked using hierarchical structure (depth levels)
        and processed in concurrent batches.

        Thread Safety:
            - CPU work runs in dedicated ThreadPoolExecutor
            - Repository saves serialized with asyncio.Lock
            - CancelledError propagated for clean shutdown

        Args:
            request: RecursiveChunkingRequest with document IDs and config

        Returns:
            RecursiveChunkingResponse with hierarchy statistics

        Raises:
            asyncio.CancelledError: If task is cancelled (propagated)
        """
        # Lazy-initialize save lock in async context (required for asyncio.Lock)
        if not isinstance(self._save_lock, asyncio.Lock):
            self._save_lock = asyncio.Lock()

        start_time = time.perf_counter()

        try:
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

            logger.info(
                f"Starting parallel hierarchical chunking for {len(documents)} documents "
                f"(max_concurrent={self._max_concurrent_chunks})"
            )

            # Process documents in batches to control concurrency
            results: List[DocumentChunkSummary] = []
            total_chunks = 0
            total_hierarchy_stats: Dict[str, int] = {}

            for batch_start in range(0, len(documents), self._max_concurrent_chunks):
                batch_end = min(batch_start + self._max_concurrent_chunks, len(documents))
                batch_documents = documents[batch_start:batch_end]

                # Create async tasks for this batch
                batch_tasks = [
                    self._chunk_and_save_document(document, chunker, request)
                    for document in batch_documents
                ]

                # Process batch in parallel
                batch_results = await asyncio.gather(
                    *batch_tasks,
                    return_exceptions=True,
                )

                # Process batch results
                for doc, result in zip(batch_documents, batch_results):
                    if isinstance(result, Exception):
                        # Handle CancelledError specially - propagate it
                        if isinstance(result, asyncio.CancelledError):
                            raise result
                        logger.error(
                            f"Failed to chunk document {doc.id}: {str(result)}"
                        )
                        results.append(
                            DocumentChunkSummary(
                                document_id=doc.id,
                                chunk_count=0,
                                hierarchy_stats={},
                                dry_run=request.dry_run,
                                warnings=[f"Chunking failed: {str(result)}"],
                            )
                        )
                    else:
                        chunk_count, hierarchy_stats, duration = result
                        total_chunks += chunk_count
                        for depth_key, count in hierarchy_stats.items():
                            total_hierarchy_stats[depth_key] = (
                                total_hierarchy_stats.get(depth_key, 0) + count
                            )
                        results.append(
                            DocumentChunkSummary(
                                document_id=doc.id,
                                chunk_count=chunk_count,
                                hierarchy_stats=hierarchy_stats,
                                dry_run=request.dry_run,
                            )
                        )
                        logger.debug(
                            f"Chunked document {doc.id}: {chunk_count} chunks in {duration:.3f}s"
                        )

            total_duration = time.perf_counter() - start_time
            logger.info(
                f"Hierarchical chunking complete: {total_chunks} chunks from {len(documents)} documents "
                f"in {total_duration:.2f}s "
                f"({total_chunks / total_duration:.1f} chunks/sec)"
            )

            return RecursiveChunkingResponse(
                success=True,
                results=results,
                total_chunks=total_chunks,
                total_hierarchy_stats=total_hierarchy_stats,
                dry_run=request.dry_run,
            )
        except asyncio.CancelledError:
            # Propagate cancellation error for clean shutdown
            logger.warning("Hierarchical chunking cancelled")
            raise
        except Exception as e:
            logger.error(f"Hierarchical chunking failed with unexpected error: {str(e)}")
            raise

    async def _chunk_and_save_document(
        self,
        document: Document,
        chunker: LlamaHierarchicalChunker,
        request: RecursiveChunkingRequest,
    ) -> tuple[int, Dict[str, int], float]:
        """Chunk a single document and save results.

        Thread Safety:
            - Chunking runs in dedicated ThreadPoolExecutor (CPU-bound work)
            - Repository saves serialized with asyncio.Lock

        Args:
            document: Document to chunk
            chunker: Configured LlamaHierarchicalChunker instance
            request: RecursiveChunkingRequest with configuration

        Returns:
            Tuple of (chunk_count, hierarchy_stats, duration_seconds)

        Raises:
            Exception: Re-raised from chunking operation
            asyncio.CancelledError: Propagated for clean cancellation
        """
        doc_start = time.perf_counter()

        # Run chunking in dedicated thread pool to avoid blocking event loop
        # (hierarchical chunking is CPU-bound)
        chunks = await asyncio.to_thread(
            self._chunk_document,
            chunker,
            document,
            request,
            executor=self._executor,
        )

        chunk_count = len(chunks)
        hierarchy_stats = self._calculate_hierarchy_stats(chunks)

        # Save if not dry run (serialize saves with asyncio.Lock)
        if not request.dry_run:
            serialized_chunks = [chunk.as_dict() for chunk in chunks]
            if request.replace_existing_chunks or not document.chunks:
                document.set_chunks(serialized_chunks)
            else:
                merged = document.chunks + serialized_chunks
                document.set_chunks(merged)

            # Serialize repository save to ensure thread safety
            async with self._save_lock:
                await self._repository.save(document)

        doc_duration = time.perf_counter() - doc_start
        return chunk_count, hierarchy_stats, doc_duration

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
