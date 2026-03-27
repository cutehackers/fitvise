"""Semantic chunking use case integrating the LlamaIndex-based chunker (Task 2.1.1).

Thread Safety Design:
- CPU-bound chunking: Dedicated ThreadPoolExecutor (via asyncio.to_thread)
- Embedding model access: Serialized with threading.Lock (not thread-safe)
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
from typing import Any, Dict, Iterable, List, Optional, Sequence
from uuid import UUID

from app.config.ml_models import get_chunking_config

logger = logging.getLogger(__name__)
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
        embedding_model: Any,
        chunker: Optional[LlamaIndexChunker] = None,
        max_concurrent_chunks: int = 4,
        thread_pool_size: int = 4,
    ) -> None:
        """Initialize semantic chunking use case.

        Args:
            document_repository: Repository for document persistence
            embedding_model: Pre-initialized HuggingFaceEmbedding model instance
            chunker: Optional pre-configured chunker. If not provided, created from config.
            max_concurrent_chunks: Max concurrent document chunks (default: 4).
                                  Controls parallelism to avoid memory overflow.
            thread_pool_size: Max threads in dedicated executor for CPU-bound work (default: 4).

        Thread Safety:
            - embedding_model access serialized with threading.Lock
            - repository.save() calls serialized with asyncio.Lock
            - CPU-bound chunking runs in dedicated ThreadPoolExecutor
            - Cancellation (CancelledError) is propagated for clean shutdown
        """
        self._repository = document_repository
        self._embedding_model = embedding_model
        self._chunker = chunker
        self._max_concurrent_chunks = max_concurrent_chunks

        # Thread safety mechanisms
        self._embedding_model_lock = threading.Lock()  # Serializes embedding model access
        self._save_lock = asyncio.Lock()  # Serializes repository saves (created lazily)
        self._executor = ThreadPoolExecutor(
            max_workers=thread_pool_size, thread_name_prefix="chunking-worker"
        )  # Dedicated pool for CPU-bound work

    async def execute(self, request: SemanticChunkingRequest) -> SemanticChunkingResponse:
        """Execute semantic chunking with parallel document processing.

        Documents are chunked in concurrent batches using asyncio.gather() to
        control parallelism and avoid memory overflow. The embedding model is
        reused across all documents for efficiency.

        Thread Safety:
            - Create save_lock in async context (required for asyncio.Lock)
            - Serialize embedding model access with threading.Lock
            - Serialize repository saves with asyncio.Lock
            - Propagate CancelledError for clean cancellation

        Args:
            request: SemanticChunkingRequest with document IDs and config

        Returns:
            SemanticChunkingResponse with results and statistics

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
                return SemanticChunkingResponse(
                    success=True,
                    results=[],
                    total_chunks=0,
                    dry_run=request.dry_run,
                )

            config_payload = request.chunker_config or get_chunking_config()
            chunker = self._chunker or LlamaIndexChunker(
                config=LlamaIndexChunkerConfig.from_dict(config_payload),
                embed_model=self._embedding_model,  # Pass injected embedding model
                require_llama_index=False,
            )

            logger.info(
                f"Starting parallel semantic chunking for {len(documents)} documents "
                f"(max_concurrent={self._max_concurrent_chunks})"
            )

            # Process documents in batches to control concurrency
            results: List[DocumentChunkSummary] = []
            total_chunks = 0

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
                                dry_run=request.dry_run,
                                warnings=[f"Chunking failed: {str(result)}"],
                            )
                        )
                    else:
                        chunk_count, duration = result
                        total_chunks += chunk_count
                        results.append(
                            DocumentChunkSummary(
                                document_id=doc.id,
                                chunk_count=chunk_count,
                                dry_run=request.dry_run,
                            )
                        )
                        logger.debug(
                            f"Chunked document {doc.id}: {chunk_count} chunks in {duration:.3f}s"
                        )

            total_duration = time.perf_counter() - start_time
            logger.info(
                f"Semantic chunking complete: {total_chunks} chunks from {len(documents)} documents "
                f"in {total_duration:.2f}s "
                f"({total_chunks / total_duration:.1f} chunks/sec)"
            )

            return SemanticChunkingResponse(
                success=True,
                results=results,
                total_chunks=total_chunks,
                dry_run=request.dry_run,
            )
        except asyncio.CancelledError:
            # Propagate cancellation error for clean shutdown
            logger.warning("Semantic chunking cancelled")
            raise
        except Exception as e:
            logger.error(f"Semantic chunking failed with unexpected error: {str(e)}")
            raise

    async def _chunk_and_save_document(
        self,
        document: Document,
        chunker: LlamaIndexChunker,
        request: SemanticChunkingRequest,
    ) -> tuple[int, float]:
        """Chunk a single document and save results.

        Thread Safety:
            - Chunking runs in dedicated ThreadPoolExecutor (CPU-bound work)
            - Embedding model access serialized with threading.Lock
            - Repository saves serialized with asyncio.Lock

        Args:
            document: Document to chunk
            chunker: Configured LlamaIndexChunker instance
            request: SemanticChunkingRequest with configuration

        Returns:
            Tuple of (chunk_count, duration_seconds)

        Raises:
            Exception: Re-raised from chunking operation
            asyncio.CancelledError: Propagated for clean cancellation
        """
        doc_start = time.perf_counter()

        # Run chunking in dedicated thread pool to avoid blocking event loop
        # (HuggingFace embedding inference is CPU-bound)
        # The embedding model access inside _chunk_document is protected by threading.Lock
        loop = asyncio.get_running_loop()
        chunks = await loop.run_in_executor(
            self._executor,
            self._chunk_document,
            chunker,
            document,
            request,
        )

        chunk_count = len(chunks)

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
        return chunk_count, doc_duration

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
        """Chunk document in thread pool context.

        Thread Safety:
            - Runs in dedicated ThreadPoolExecutor (CPU-bound work)
            - Embedding model access protected by threading.Lock (not thread-safe)
            - This is a synchronous function that runs in thread pool

        Args:
            chunker: LlamaIndexChunker instance
            document: Document to chunk
            request: SemanticChunkingRequest with configuration

        Returns:
            List of Chunk objects
        """
        text_candidates: Iterable[Optional[str]] = (
            document.extracted_text,
            document.content,
        )
        text = next((value for value in text_candidates if value), "")

        metadata_payload = _default_chunk_metadata(document)
        if request.metadata_overrides:
            metadata_payload.update(request.metadata_overrides)

        document_metadata = _document_metadata_dict(document)

        # Serialize embedding model access (HuggingFaceEmbedding is not thread-safe)
        # The chunker uses self._embedding_model internally, so we must protect it
        with self._embedding_model_lock:
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
