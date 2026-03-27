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

from app.di.containers.container import AppContainer
from app.di.containers.infra_container import InfraContainer
from app.pipeline.config import PipelineSpec
from app.pipeline.chunking_config_resolver import resolve_chunking_configuration
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
from app.domain.entities.chunk_load_policy import ChunkLoadPolicy
from app.domain.entities.chunk_load_policy import ChunkLoadPolicy
from app.domain.repositories.document_repository import DocumentRepository
from app.domain.repositories.embedding_repository import EmbeddingRepository

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
        infra: InfraContainer,
        document_repository: DocumentRepository,
        embedding_repository: EmbeddingRepository,
        verbose: bool = False,
    ):
        """Initialize the embedding phase.

        Args:
            infra: Infrastructure container with all required services (ML models, embedding services, etc.)
            document_repository: Shared document repository instance
            verbose: Enable verbose logging
        """
        self.document_repository = document_repository
        self.embedding_repository = embedding_repository
        self.infra = infra
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
            # Get all processed documents from the SHARED repository
            documents = await self.document_repository.find_processed_documents()

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

    async def _validate_chunk_availability(
        self, document_ids: List[UUID]
    ) -> Dict[str, Any]:
        """Validate that chunks exist for all documents from Task 2 (Ingestion).

        This method verifies the handover from Task 2 by checking that all documents
        have been properly chunked and are ready for embedding generation.

        Args:
            document_ids: List of document IDs to validate

        Returns:
            Dict with chunk statistics:
                - total_documents: Total number of documents checked
                - documents_with_chunks: Number of documents that have chunks
                - total_chunks: Total number of chunks available
                - documents_without_chunks: List of document IDs missing chunks

        Raises:
            EmbeddingPipelineError: If chunk validation fails critically
        """
        logger.info("🔍 Validating chunk availability from Task 2 (Ingestion)...")

        chunk_stats = {
            "total_documents": len(document_ids),
            "documents_with_chunks": 0,
            "total_chunks": 0,
            "documents_without_chunks": []
        }

        try:
            for doc_id in document_ids:
                # Get document from repository and access chunks property
                doc = await self.document_repository.find_by_id(doc_id)

                # Check if document exists and has chunks
                if doc and hasattr(doc, 'chunks') and doc.chunks:
                    chunks = doc.chunks
                    chunk_stats["documents_with_chunks"] += 1
                    chunk_stats["total_chunks"] += len(chunks)
                    if self.verbose:
                        logger.debug(f"✅ Document {doc_id}: {len(chunks)} chunks found")
                else:
                    chunk_stats["documents_without_chunks"].append(str(doc_id))
                    logger.warning(f"⚠️ Document {doc_id}: No chunks found")

            return chunk_stats

        except Exception as e:
            logger.error(f"Failed to validate chunk availability: {str(e)}")
            raise EmbeddingPipelineError(f"Chunk validation failed: {str(e)}")

    async def execute(
        self,
        spec: PipelineSpec,
        batch_size: int = 32,
        deduplication_enabled: bool = True,
        max_retries: int = 3,
        show_progress: bool = True,
        document_limit: Optional[int] = None,
        chunk_load_policy: ChunkLoadPolicy = ChunkLoadPolicy.EXISTING_ONLY,
    ) -> RagEmbeddingTaskReport:
        """Execute embedding generation phase.

        This phase (Task 3) assumes that documents have already been chunked by Task 2 (Ingestion).
        It validates chunk availability, then generates and stores embeddings for existing chunks.

        Args:
            spec: Pipeline specification
            batch_size: Batch size for embedding generation
            deduplication_enabled: Whether to enable chunk deduplication
            max_retries: Maximum retry attempts for failed operations
            show_progress: Whether to show progress during processing
            document_limit: Optional limit on documents to process
            chunk_load_policy: Policy for loading chunks from Task 2.
                             - EXISTING_ONLY (default): Strict mode - fail if chunks missing
                             - SENTENCE_FALLBACK: Use sentence splitter as fallback
                             - SEMANTIC_FALLBACK: Use semantic splitter as fallback
                             Default is EXISTING_ONLY for production-safe fail-fast behavior.

        Returns:
            RagEmbeddingTaskReport with comprehensive statistics and timing

        Raises:
            EmbeddingPipelineError: If pipeline execution fails
        """
        from datetime import timezone
        phase_start_time = datetime.now(timezone.utc)
        start_time = time.time()

        logger.info("Starting RAG embedding pipeline (Task 3: Embedding Generation)...")
        if self.verbose:
            logger.info(
                f"Configuration: batch_size={batch_size}, deduplication={deduplication_enabled}, max_retries={max_retries}"
            )

        try:
            # Initialize use cases with SHARED repository
            logger.info("Initializing embedding use cases...")

            # Step 1: Ensure Weaviate client is connected
            logger.info("🔗 Ensuring Weaviate client connection...")
            try:
                # Note: InfraContainer doesn't have ensure_weaviate_connected method
                # Weaviate client is already connected in build_llama_index_resources
                logger.info("✅ Weaviate client connection handled by InfraContainer")
            except Exception as weaviate_error:
                error_msg = f"Failed to connect Weaviate client: {str(weaviate_error)}"
                logger.error(error_msg)
                raise EmbeddingPipelineError(error_msg) from weaviate_error

            # Step 2: Ensure embedding service is initialized before use
            embedding_service = self.infra.sentence_transformer_service()
            logger.info(f"Initializing embedding model: {embedding_service.model_name}")
            try:
                await embedding_service.initialize()
                logger.info("Embedding model initialized successfully")
            except Exception as init_error:
                error_msg = f"Failed to initialize embedding model '{embedding_service.model_name}': {str(init_error)}"
                logger.error(error_msg)
                raise EmbeddingPipelineError(error_msg) from init_error

            # Embedding use case with injected services
            embedding_use_case = EmbedDocumentChunksUseCase(
                embedding_service=embedding_service,
                embedding_repository=self.embedding_repository,
            )

            # Build ingestion pipeline use case with shared repository
            # Note: This still requires chunking_use_case for BuildIngestionPipelineUseCase interface
            # but chunks will be loaded from repository, not re-chunked
            chunking_use_case = SemanticChunkingUseCase(
                document_repository=self.document_repository,
                embedding_model=self.infra.embedding
            )

            build_use_case = BuildIngestionPipelineUseCase(
                document_repository=self.document_repository,
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

            # Validate chunk handover from Task 2 (Ingestion)
            logger.info("=" * 80)
            logger.info("📦 CHUNK HANDOVER VALIDATION (Task 2 → Task 3)")
            logger.info("=" * 80)

            chunk_stats = await self._validate_chunk_availability(document_ids)

            logger.info("📊 Chunk Statistics:")
            logger.info(f"   Documents processed: {chunk_stats['total_documents']}")
            logger.info(f"   Documents with chunks: {chunk_stats['documents_with_chunks']}")
            logger.info(f"   Total chunks available: {chunk_stats['total_chunks']}")

            if chunk_stats['documents_without_chunks']:
                logger.warning(
                    f"⚠️ {len(chunk_stats['documents_without_chunks'])} documents have no chunks"
                )
                if self.verbose:
                    logger.warning(
                        f"   Missing chunk documents (first 5): {chunk_stats['documents_without_chunks'][:5]}"
                    )

            if chunk_stats['total_chunks'] == 0:
                logger.error("❌ No chunks available for embedding generation")
                logger.error("   Possible causes:")
                logger.error("   1. Task 2 (Ingestion) not completed")
                logger.error("   2. Chunking failed during Task 2")
                logger.error("   3. Repository connection issue")

                # Return early with clear error
                from datetime import timezone
                phase_end_time = datetime.now(timezone.utc)
                execution_time = (phase_end_time - phase_start_time).total_seconds()

                error_result = EmbeddingResult(
                    success=False,
                    documents_processed=len(document_ids),
                    total_chunks=0,
                    unique_chunks=0,
                    duplicates_removed=0,
                    embeddings_generated=0,
                    embeddings_stored=0,
                    processing_time_seconds=time.time() - start_time,
                    errors=["No chunks available for embedding generation - Task 2 may not have completed"],
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

            logger.info(f"✅ Chunk handover validated: {chunk_stats['total_chunks']} chunks ready")
            logger.info("🚀 Proceeding to embedding generation...")
            logger.info("=" * 80)

            # Resolve chunking configuration (single source of truth for semantic chunking)
            resolved_chunker_config = resolve_chunking_configuration(spec)
            enable_semantic_chunking = resolved_chunker_config.get("enable_semantic_chunking", True)
            logger.info(f"📋 Resolved Chunking Configuration:")
            logger.info(f"   Method: {'semantic' if enable_semantic_chunking else 'sentence'}")
            logger.info(f"   Source: {resolved_chunker_config.get('preset', 'default')}")

            # Create pipeline request with resolved configuration
            # Note: Chunks will be loaded from repository (created in Task 2)
            pipeline_request = BuildIngestionPipelineRequest(
                document_ids=document_ids,
                batch_size=batch_size,
                deduplication_enabled=deduplication_enabled,
                max_retries=max_retries,
                show_progress=show_progress,
                replace_existing_embeddings=False,  # Incremental approach
                chunk_load_policy=chunk_load_policy,  # Policy for loading chunks from Task 2
                chunker_config=resolved_chunker_config,  # Single source of truth (from spec resolution)
            )

            logger.info(f"📝 Processing Configuration:")
            logger.info(f"   Documents: {len(document_ids)}")
            logger.info(f"   Available chunks: {chunk_stats['total_chunks']}")
            logger.info(f"   Batch size: {batch_size}")
            logger.info(f"   Deduplication: {deduplication_enabled}")
            logger.info(f"   Max retries: {max_retries}")
            logger.info(f"   Chunk load policy: {chunk_load_policy} ({str(chunk_load_policy)})")
            logger.info(f"   Chunking method: {'semantic (with embeddings)' if enable_semantic_chunking else 'sentence (no embeddings)'}")

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
