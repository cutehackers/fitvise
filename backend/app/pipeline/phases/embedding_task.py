"""Embedding Generation Phase.

This phase generates embeddings for processed document chunks and stores
them in the vector database.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from app.pipeline.config import PipelineSpec
from app.application.use_cases.indexing.build_ingestion_pipeline import (
    BuildIngestionPipelineUseCase,
    BuildIngestionPipelineRequest,
)
from app.application.use_cases.chunking.semantic_chunking import (
    SemanticChunkingUseCase,
)
from app.application.use_cases.embedding.embed_document_chunks import (
    EmbedDocumentChunksUseCase,
)
from app.domain.repositories.document_repository import DocumentRepository

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding pipeline execution."""

    success: bool
    documents_processed: int
    total_chunks: int
    unique_chunks: int
    duplicates_removed: int
    embeddings_generated: int
    embeddings_stored: int
    processing_time_seconds: float
    deduplication_stats: Optional[Dict[str, Any]] = None
    processing_stats: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "documents_processed": self.documents_processed,
            "total_chunks": self.total_chunks,
            "unique_chunks": self.unique_chunks,
            "duplicates_removed": self.duplicates_removed,
            "duplicates_percentage": (
                round((self.duplicates_removed / self.total_chunks * 100), 2)
                if self.total_chunks > 0
                else 0
            ),
            "embeddings_generated": self.embeddings_generated,
            "embeddings_stored": self.embeddings_stored,
            "processing_time_seconds": round(self.processing_time_seconds, 2),
            "deduplication_stats": self.deduplication_stats,
            "processing_stats": self.processing_stats,
            "errors": self.errors,
            "warnings": self.warnings,
            "chunks_per_document": (
                round(self.total_chunks / self.documents_processed, 2)
                if self.documents_processed > 0
                else 0
            ),
            "embedding_success_rate": (
                round((self.embeddings_stored / self.unique_chunks * 100), 2)
                if self.unique_chunks > 0
                else 0
            ),
        }


class EmbeddingPipelineError(Exception):
    """Embedding pipeline execution failed."""

    pass


@dataclass
class RagEmbeddingTaskReport:
    """Report for embedding generation and storage task execution.

    Wraps EmbeddingResult with timing metadata.
    """

    success: bool = False
    execution_time_seconds: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    phase_result: Optional[EmbeddingResult] = None
    total_errors: int = 0
    total_warnings: int = 0

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result_dict = {}
        if self.phase_result and hasattr(self.phase_result, 'as_dict'):
            result_dict = self.phase_result.as_dict()
        elif self.phase_result:
            result_dict = self.phase_result.__dict__

        return {
            "task_name": "Embedding Generation",
            "success": self.success,
            "execution_time_seconds": round(self.execution_time_seconds, 2),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "phase_result": result_dict,
            "total_errors": self.total_errors,
            "total_warnings": self.total_warnings
        }

    def as_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.as_dict(), indent=indent, default=str)


class RagEmbeddingTask:
    """Task 3: Embedding Generation and Storage.

    Generates embeddings for processed document chunks and stores them
    in the vector database with deduplication.
    """

    def __init__(
        self,
        document_repository: DocumentRepository,
        verbose: bool = False,
    ):
        """Initialize the embedding phase.

        Args:
            document_repository: Shared document repository instance
            verbose: Enable verbose logging
        """
        self.document_repository = document_repository
        self.verbose = verbose

        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

    async def _get_processed_document_ids(
        self, limit: Optional[int] = None
    ) -> List[UUID]:
        """Get IDs of processed documents from shared repository.

        Args:
            limit: Optional limit on number of documents to process

        Returns:
            List of document IDs

        Raises:
            EmbeddingPipelineError: If document retrieval fails
        """
        try:
            # Get all documents from the SHARED repository
            documents = self.document_repository.list_all()

            # Convert to document IDs
            document_ids = (
                [doc.id for doc in documents[:limit]]
                if limit
                else [doc.id for doc in documents]
            )

            if not document_ids:
                logger.warning("No processed documents found for embedding generation")

            logger.info(
                f"Found {len(document_ids)} processed documents for embedding"
            )
            return document_ids

        except Exception as e:
            logger.error(f"Failed to get processed document IDs: {str(e)}")
            raise EmbeddingPipelineError(f"Document retrieval failed: {str(e)}")

    async def execute(
        self,
        spec: PipelineSpec,
        batch_size: int = 32,
        deduplication_enabled: bool = True,
        max_retries: int = 3,
        show_progress: bool = True,
        document_limit: Optional[int] = None,
        skip_chunking: bool = True,
    ) -> RagEmbeddingTaskReport:
        """Execute embedding generation phase.

        Args:
            spec: Pipeline specification
            batch_size: Batch size for embedding generation
            deduplication_enabled: Whether to enable chunk deduplication
            max_retries: Maximum retry attempts for failed operations
            show_progress: Whether to show progress during processing
            document_limit: Optional limit on documents to process
            skip_chunking: If True, load existing chunks from repository instead of re-chunking.
                          Defaults to True for optimization (reuse chunks from Task 2).
                          Set to False to force re-chunking with current configuration.

        Returns:
            RagEmbeddingTaskReport with comprehensive statistics and timing

        Raises:
            EmbeddingPipelineError: If pipeline execution fails
        """
        from datetime import timezone
        phase_start_time = datetime.now(timezone.utc)
        start_time = time.time()

        logger.info("Starting RAG embedding pipeline...")
        if self.verbose:
            logger.info(
                f"Configuration: batch_size={batch_size}, deduplication={deduplication_enabled}, max_retries={max_retries}"
            )

        try:
            # Initialize use cases with SHARED repository
            logger.info("Initializing use cases with shared repository...")

            # Chunking use case with shared repository
            chunking_use_case = SemanticChunkingUseCase(
                document_repository=self.document_repository
            )

            # Embedding use case
            embedding_use_case = EmbedDocumentChunksUseCase()

            # Build ingestion pipeline use case with shared repository
            build_use_case = BuildIngestionPipelineUseCase(
                document_repository=self.document_repository,  # Use shared repository!
                chunking_use_case=chunking_use_case,
                embedding_use_case=embedding_use_case,
            )

            # Get document IDs from shared repository
            document_ids = await self._get_processed_document_ids(
                limit=document_limit
            )

            if not document_ids:
                logger.warning("No documents to process")
                from datetime import timezone
                phase_end_time = datetime.now(timezone.utc)
                execution_time = (phase_end_time - phase_start_time).total_seconds()

                empty_result = EmbeddingResult(
                    success=True,
                    documents_processed=0,
                    total_chunks=0,
                    unique_chunks=0,
                    duplicates_removed=0,
                    embeddings_generated=0,
                    embeddings_stored=0,
                    processing_time_seconds=time.time() - start_time,
                    warnings=["No documents found for embedding generation"],
                )

                return RagEmbeddingTaskReport(
                    success=True,
                    execution_time_seconds=execution_time,
                    start_time=phase_start_time,
                    end_time=phase_end_time,
                    phase_result=empty_result,
                    total_errors=0,
                    total_warnings=1,
                )

            # Create pipeline request
            pipeline_request = BuildIngestionPipelineRequest(
                document_ids=document_ids,
                batch_size=batch_size,
                deduplication_enabled=deduplication_enabled,
                max_retries=max_retries,
                show_progress=show_progress,
                replace_existing_embeddings=False,  # Incremental approach
                skip_chunking=skip_chunking,  # Use parameter value (default True for optimization)
            )

            if skip_chunking:
                logger.info(f"Processing {len(document_ids)} documents with skip_chunking=True")
                logger.info("Chunks will be loaded from repository (created in Task 2)")
            else:
                logger.info(f"Processing {len(document_ids)} documents with skip_chunking=False")
                logger.info("Documents will be re-chunked with current configuration")

            # Execute pipeline
            pipeline_response = await build_use_case.execute(pipeline_request)

            processing_time = time.time() - start_time

            # Build result
            result = EmbeddingResult(
                success=pipeline_response.success,
                documents_processed=len(document_ids),
                total_chunks=(
                    pipeline_response.deduplication_stats.total_chunks
                    if pipeline_response.deduplication_stats
                    else 0
                ),
                unique_chunks=(
                    pipeline_response.deduplication_stats.unique_chunks
                    if pipeline_response.deduplication_stats
                    else 0
                ),
                duplicates_removed=(
                    pipeline_response.deduplication_stats.duplicates_removed
                    if pipeline_response.deduplication_stats
                    else 0
                ),
                embeddings_generated=pipeline_response.results.get(
                    "embeddings_generated", 0
                ),
                embeddings_stored=pipeline_response.results.get(
                    "embeddings_stored", 0
                ),
                processing_time_seconds=processing_time,
                deduplication_stats=(
                    pipeline_response.deduplication_stats.as_dict()
                    if pipeline_response.deduplication_stats
                    else None
                ),
                processing_stats=pipeline_response.processing_stats,
                errors=pipeline_response.errors,
                warnings=[],
            )

            if pipeline_response.success:
                logger.info("🎉 Embedding pipeline completed successfully!")
                logger.info(f"Documents: {result.documents_processed}")
                logger.info(
                    f"Chunks: {result.total_chunks} total, {result.unique_chunks} unique"
                )
                logger.info(f"Duplicates removed: {result.duplicates_removed}")
                logger.info(
                    f"Embeddings: {result.embeddings_generated} generated, {result.embeddings_stored} stored"
                )
                logger.info(f"Processing time: {result.processing_time_seconds:.2f}s")
            else:
                logger.error("❌ Embedding pipeline failed")
                for error in pipeline_response.errors:
                    logger.error(f"   - {error}")

            from datetime import timezone
            phase_end_time = datetime.now(timezone.utc)
            execution_time = (phase_end_time - phase_start_time).total_seconds()

            return RagEmbeddingTaskReport(
                success=result.success,
                execution_time_seconds=execution_time,
                start_time=phase_start_time,
                end_time=phase_end_time,
                phase_result=result,
                total_errors=len(result.errors),
                total_warnings=len(result.warnings),
            )

        except Exception as e:
            from datetime import timezone
            processing_time = time.time() - start_time
            phase_end_time = datetime.now(timezone.utc)
            execution_time = (phase_end_time - phase_start_time).total_seconds()
            error_msg = f"Embedding pipeline failed: {str(e)}"
            logger.error(error_msg)

            error_result = EmbeddingResult(
                success=False,
                documents_processed=0,
                total_chunks=0,
                unique_chunks=0,
                duplicates_removed=0,
                embeddings_generated=0,
                embeddings_stored=0,
                processing_time_seconds=processing_time,
                errors=[error_msg],
            )

            return RagEmbeddingTaskReport(
                success=False,
                execution_time_seconds=execution_time,
                start_time=phase_start_time,
                end_time=phase_end_time,
                phase_result=error_result,
                total_errors=1,
                total_warnings=0,
            )
