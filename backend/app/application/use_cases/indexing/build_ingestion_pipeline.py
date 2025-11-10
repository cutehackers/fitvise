"""Build embedding ingestion pipeline use case (Task 2.3.3).

This use case orchestrates the end-to-end embedding ingestion pipeline:
document chunks → deduplication → embedding generation → Weaviate storage.

Follows existing error handling patterns from setup_embedding_infrastructure.py
and integrates with existing chunking and embedding services.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from app.domain.entities.chunk import Chunk
from app.domain.entities.processing_job import (
    JobStatus,
    JobType,
    ProcessingJob,
)
from app.domain.entities.chunk_load_policy import ChunkLoadPolicy
from app.domain.exceptions.chunking_exceptions import ChunkingError
from app.domain.exceptions.embedding_exceptions import (
    DeduplicationError,
    EmbeddingGenerationError,
    EmbeddingStorageError,
    IngestionPipelineError,
)
from app.domain.repositories.document_repository import DocumentRepository
from app.application.use_cases.chunking.semantic_chunking import (
    SemanticChunkingRequest,
    SemanticChunkingUseCase,
    SemanticChunkingResponse,
)
from app.application.use_cases.embedding.embed_document_chunks import (
    EmbedChunksRequest,
    EmbedDocumentChunksUseCase,
    EmbedChunksResponse,
)

logger = logging.getLogger(__name__)


@dataclass
class BuildIngestionPipelineRequest:
    """Request for building embedding ingestion pipeline."""

    document_ids: Optional[List[UUID]] = None
    batch_size: int = 32
    deduplication_enabled: bool = True
    max_retries: int = 3
    retry_backoff_factor: float = 1.0
    show_progress: bool = True
    replace_existing_embeddings: bool = False
    chunk_load_policy: ChunkLoadPolicy = ChunkLoadPolicy.EXISTING_ONLY  # Policy for loading chunks from Task 2


@dataclass
class DeduplicationStats:
    """Statistics from content deduplication."""

    total_chunks: int
    unique_chunks: int
    duplicates_removed: int
    duplicates_percentage: float

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_chunks": self.total_chunks,
            "unique_chunks": self.unique_chunks,
            "duplicates_removed": self.duplicates_removed,
            "duplicates_percentage": round(self.duplicates_percentage, 2),
        }


@dataclass
class BuildIngestionPipelineResponse:
    """Response from embedding ingestion pipeline."""

    success: bool
    job_id: Optional[UUID] = None
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    deduplication_stats: Optional[DeduplicationStats] = None
    processing_stats: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "job_id": str(self.job_id) if self.job_id else None,
            "results": self.results,
            "errors": self.errors,
            "deduplication_stats": self.deduplication_stats.as_dict() if self.deduplication_stats else None,
            "processing_stats": self.processing_stats,
        }


class BuildIngestionPipelineUseCase:
    """Use case for building embedding ingestion pipeline (Task 2.3.3).

    Orchestrates the embedding generation workflow for Task 3:
    1. Load existing chunks from repository (created by Task 2)
    2. Content deduplication (SHA256-based)
    3. Embedding generation (reuses existing EmbedDocumentChunksUseCase)
    4. Vector storage (handled by embedding use case)

    **Architecture Note**:
    - Primary flow: Loads chunks from repository (ChunkLoadPolicy.EXISTING_ONLY)
    - Fallback optional: Re-chunks via SemanticChunkingUseCase if chunks missing (explicit opt-in)
    - Task 2 (Ingestion) is responsible for all primary chunking operations
    - Task 3 (Embedding) should always find existing chunks from Task 2

    **Chunk Load Policies**:
    - EXISTING_ONLY (default): Strict mode - fails immediately if chunks missing
    - SENTENCE_FALLBACK: Fallback to sentence splitter if chunks missing
    - SEMANTIC_FALLBACK: Fallback to semantic splitter if chunks missing

    Pipeline Flow:

     Repository Chunks → SHA256 Deduplication → Embedding Generation → Weaviate Storage
            ↓                   ↓                      ↓                      ↓
       Validation          Dedup Stats           Retry Logic           Job Completion

     Fallback Path (only with SENTENCE_FALLBACK or SEMANTIC_FALLBACK):
     Documents → Semantic Chunking (fallback) → continues with normal flow

    Examples:
        >>> use_case = BuildIngestionPipelineUseCase(
        ...     document_repository=document_repo,
        ...     chunking_use_case=chunking_use_case,  # Used only as fallback
        ...     embedding_use_case=embedding_use_case
        ... )
        >>> request = BuildIngestionPipelineRequest(
        ...     document_ids=[uuid4(), uuid4()],
        ...     chunk_load_policy=ChunkLoadPolicy.EXISTING_ONLY  # Default - strict mode
        ... )
        >>> response = await use_case.execute(request)
        >>> response.success
        True
    """

    def __init__(
        self,
        document_repository: DocumentRepository,
        chunking_use_case: SemanticChunkingUseCase,
        embedding_use_case: EmbedDocumentChunksUseCase,
    ) -> None:
        """Initialize ingestion pipeline use case.

        Args:
            document_repository: Repository for document operations
            chunking_use_case: Use case for semantic chunking (used as fallback only when
                              chunks from Task 2 are missing - should rarely be needed)
            embedding_use_case: Use case for embedding generation and storage
        """
        self._document_repository = document_repository
        self._chunking_use_case = chunking_use_case
        self._embedding_use_case = embedding_use_case
        self._current_job: Optional[ProcessingJob] = None

    async def execute(self, request: BuildIngestionPipelineRequest) -> BuildIngestionPipelineResponse:
        """Execute embedding ingestion pipeline.

        Args:
            request: Pipeline configuration and parameters

        Returns:
            Pipeline execution results with comprehensive statistics

        Raises:
            IngestionPipelineError: If pipeline initialization fails
        """
        # Initialize job for progress tracking
        self._current_job = ProcessingJob(
            job_type=JobType.EMBEDDING_GENERATION,
            name="Document Embedding Ingestion",
            description="Process documents into embeddings and store in vector database"
        )

        errors = []
        results = {}
        deduplication_stats = None

        try:
            logger.info(f"Starting embedding ingestion pipeline for {len(request.document_ids or [])} documents")
            self._current_job.start(total_steps=4)

            # Early validation for empty document IDs
            if not request.document_ids:
                self._current_job.fail("No document IDs provided")
                raise IngestionPipelineError(
                    message="No document IDs provided",
                    stage="validation",
                    document_count=0
                )

            # Step 1: Get chunks from documents
            try:
                self._current_job.update_progress(25, "Generating semantic chunks...")
                chunks = await self._get_chunks(request)
                results["chunks_generated"] = len(chunks)
                logger.info(f"Generated {len(chunks)} chunks from documents")
            except ChunkingError as e:
                error_msg = f"Chunking failed: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
                raise IngestionPipelineError(
                    message="Document chunking stage failed",
                    stage="chunking",
                    document_count=len(request.document_ids or []),
                    details=str(e)
                )
            except Exception as e:
                error_msg = f"Unexpected chunking error: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
                raise IngestionPipelineError(
                    message="Unexpected error in chunking stage",
                    stage="chunking",
                    details=str(e)
                )

            if not chunks:
                logger.warning("No chunks generated from documents")
                return BuildIngestionPipelineResponse(
                    success=True,
                    job_id=self._current_job.id,
                    results={"chunks_generated": 0},
                    errors=["No chunks generated from provided documents"],
                    processing_stats=self._current_job.get_progress_dict()
                )

            # Step 2: Deduplicate chunks
            try:
                self._current_job.update_progress(50, "Deduplicating content...")
                unique_chunks, deduplication_stats = await self._deduplicate_chunks(
                    chunks, request.deduplication_enabled
                )
                results["deduplication_stats"] = deduplication_stats.as_dict()
                logger.info(f"Deduplication removed {deduplication_stats.duplicates_removed} duplicates")
            except DeduplicationError as e:
                error_msg = f"Deduplication failed: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
                # Continue with original chunks as fallback
                unique_chunks = chunks
                deduplication_stats = DeduplicationStats(
                    total_chunks=len(chunks),
                    unique_chunks=len(chunks),
                    duplicates_removed=0,
                    duplicates_percentage=0.0
                )
                results["deduplication_stats"] = deduplication_stats.as_dict()
            except Exception as e:
                error_msg = f"Unexpected deduplication error: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
                # Continue with original chunks as fallback
                unique_chunks = chunks

            # Step 3: Generate embeddings
            try:
                self._current_job.update_progress(75, "Generating embeddings...")
                embedding_result = await self._retry_operation(
                    operation=self._generate_embeddings,
                    max_retries=request.max_retries,
                    backoff_factor=request.retry_backoff_factor,
                    chunks=unique_chunks,
                    request=request
                )
                results["embeddings_generated"] = embedding_result.embedded_count
                results["embeddings_stored"] = embedding_result.stored_count
                logger.info(f"Generated {embedding_result.embedded_count} embeddings, stored {embedding_result.stored_count}")
            except EmbeddingGenerationError as e:
                error_msg = f"Embedding generation failed: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
                raise IngestionPipelineError(
                    message="Embedding generation stage failed",
                    stage="embedding",
                    chunks_processed=len(unique_chunks),
                    details=str(e)
                )
            except Exception as e:
                error_msg = f"Unexpected embedding error: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
                raise IngestionPipelineError(
                    message="Unexpected error in embedding stage",
                    stage="embedding",
                    details=str(e)
                )

            # Complete pipeline successfully
            self._current_job.update_progress(100, "Pipeline completed successfully")
            self._current_job.complete({
                "chunks_generated": results.get("chunks_generated", 0),
                "unique_chunks": deduplication_stats.unique_chunks if deduplication_stats else 0,
                "embeddings_generated": results.get("embeddings_generated", 0),
                "embeddings_stored": results.get("embeddings_stored", 0),
                "duplicates_removed": deduplication_stats.duplicates_removed if deduplication_stats else 0,
                "errors": errors
            })

            logger.info(f"Pipeline completed successfully. Job ID: {self._current_job.id}")

            return BuildIngestionPipelineResponse(
                success=True,
                job_id=self._current_job.id,
                results=results,
                errors=errors,
                deduplication_stats=deduplication_stats,
                processing_stats=self._current_job.get_progress_dict()
            )

        except Exception as e:
            # Mark job as failed if not already failed
            if self._current_job and self._current_job.status != JobStatus.FAILED:
                self._current_job.fail(str(e))

            logger.error(f"Pipeline failed: {str(e)}")

            # Return failure response
            return BuildIngestionPipelineResponse(
                success=False,
                job_id=self._current_job.id if self._current_job else None,
                results=results,
                errors=errors + [str(e)],
                deduplication_stats=deduplication_stats,
                processing_stats=self._current_job.get_progress_dict() if self._current_job else {}
            )

    async def _load_chunks(self, document_ids: List[UUID]) -> List[Chunk]:
        """Load chunks that were already created by ingestion task.

        This method retrieves pre-existing chunks from the document repository,
        avoiding the need to re-run semantic chunking.

        Args:
            document_ids: List of document IDs to load chunks from

        Returns:
            List of Chunk objects from document repository

        Raises:
            ChunkingError: If chunk loading encounters critical errors
        """
        all_chunks = []
        failed_loads = 0
        conversion_errors = 0
        total_chunks_processed = 0

        for doc_id in document_ids:
            try:
                doc = await self._document_repository.find_by_id(doc_id)
                if doc and hasattr(doc, 'chunks') and doc.chunks:
                    doc_chunks_count = len(doc.chunks)
                    logger.debug(f"Document {doc_id}: Found {doc_chunks_count} serialized chunks")
                    total_chunks_processed += doc_chunks_count

                    # Convert stored chunk dictionaries to Chunk objects
                    for i, chunk_data in enumerate(doc.chunks):
                        try:
                            # Use Chunk.from_dict method for proper deserialization
                            if hasattr(Chunk, 'from_dict'):
                                chunk = Chunk.from_dict(chunk_data)
                                all_chunks.append(chunk)
                            else:
                                # Fallback: direct instantiation from dict
                                chunk = Chunk(**chunk_data)
                                all_chunks.append(chunk)
                        except Exception as e:
                            conversion_errors += 1
                            logger.error(f"Failed to convert chunk {i} from document {doc_id}: {e}")
                            logger.debug(f"Chunk data keys: {list(chunk_data.keys()) if isinstance(chunk_data, dict) else type(chunk_data)}")
                            continue
                else:
                    logger.warning(f"Document {doc_id}: No chunks property or empty chunks")
            except Exception as e:
                logger.error(f"Failed to load document {doc_id}: {e}")
                failed_loads += 1
                continue

        if conversion_errors > 0:
            logger.error(f"❌ Failed to convert {conversion_errors} chunks due to deserialization errors")
            logger.error("   This indicates a serialization/deserialization mismatch between Task 2 and Task 3")

        if failed_loads > 0:
            logger.warning(f"Failed to load chunks from {failed_loads} documents")

        success_rate = (len(all_chunks) / total_chunks_processed * 100) if total_chunks_processed > 0 else 0
        logger.info(f"✅ Loaded {len(all_chunks)} existing chunks from {len(document_ids) - failed_loads}/{len(document_ids)} documents")
        logger.info(f"   Conversion success rate: {success_rate:.1f}% ({len(all_chunks)}/{total_chunks_processed})")

        return all_chunks

    async def _get_chunks(self, request: BuildIngestionPipelineRequest) -> List[Chunk]:
        """Get chunks from documents based on chunk load policy.

        **Chunk Load Policies**:
        - EXISTING_ONLY (default): Load existing chunks only, fail if missing
        - SENTENCE_FALLBACK: Load existing or fallback to sentence splitter
        - SEMANTIC_FALLBACK: Load existing or fallback to semantic splitter

        **Primary Path (Task 3 - Embedding Generation)**:
        1. Load existing chunks from repository (created by Task 2)
        2. Chunks should always exist if Task 2 completed successfully

        **Fallback Path (only with SENTENCE_FALLBACK or SEMANTIC_FALLBACK)**:
        1. If no existing chunks found → Generate new chunks via SemanticChunkingUseCase
        2. This indicates Task 2 may not have completed properly

        Args:
            request: Pipeline request with document IDs and chunk load policy

        Returns:
            List of chunks (either loaded or generated)

        Raises:
            ChunkingError: If chunks are missing and policy is EXISTING_ONLY, or if fallback chunking fails
        """
        if not request.document_ids:
            raise ChunkingError("No document IDs provided for chunking")

        # Always try loading existing chunks first
        logger.info(f"📥 Loading existing chunks from repository (Task 2 → Task 3 handover)")
        logger.info(f"   Policy: {request.chunk_load_policy}")
        chunks = await self._load_chunks(request.document_ids)

        if chunks:
            logger.info(f"✅ Successfully loaded {len(chunks)} existing chunks from Task 2 (Ingestion)")
            return chunks

        # Handle missing chunks based on policy
        logger.warning("=" * 80)
        logger.warning("⚠️  WARNING: No existing chunks found in repository!")
        logger.warning("   This indicates Task 2 (Ingestion) may not have completed successfully")
        logger.warning(f"   Current policy: {request.chunk_load_policy}")

        if request.chunk_load_policy == ChunkLoadPolicy.EXISTING_ONLY:
            # Strict mode - fail immediately
            logger.error("   Policy is EXISTING_ONLY - failing immediately (no fallback allowed)")
            logger.error("   Recommendation: Verify Task 2 completed successfully and re-run if needed")
            logger.warning("=" * 80)
            raise ChunkingError(
                "No existing chunks found and chunk_load_policy is EXISTING_ONLY. "
                "Task 2 (Ingestion) must complete successfully before Task 3 (Embedding). "
                "Please run Task 2 first or change chunk_load_policy to allow fallback.",
                document_id=str(request.document_ids)
            )

        # Fallback mode - generate new chunks
        if not request.chunk_load_policy.requires_fallback():
            # Should not reach here, but safety check
            logger.error("   Invalid policy state - no fallback allowed")
            logger.warning("=" * 80)
            raise ChunkingError(
                f"Invalid chunk load policy: {request.chunk_load_policy}",
                document_id=str(request.document_ids)
            )

        # Determine chunking method based on policy
        use_semantic = request.chunk_load_policy.uses_semantic_chunking()
        chunking_method = "semantic (with embeddings)" if use_semantic else "sentence (no embeddings)"
        logger.warning(f"   Falling back to re-chunking using: {chunking_method}")
        logger.warning("   Recommendation: Verify Task 2 completed and re-run if needed")
        logger.warning("=" * 80)

        logger.info(f"🔄 Generating new chunks using SemanticChunkingUseCase (fallback mode: {chunking_method})")
        chunking_request = SemanticChunkingRequest(
            document_ids=request.document_ids,
            include_failed=False,  # Don't include failed chunks in pipeline
            replace_existing_chunks=True,
            enable_semantic=use_semantic  # Use chunking method specified by policy
        )

        chunking_response = await self._chunking_use_case.execute(chunking_request)

        if not chunking_response.success:
            raise ChunkingError(
                f"Fallback chunking failed - Task 2 chunks missing and re-chunking unsuccessful",
                document_id=str(request.document_ids)
            )

        # Load chunks from documents after chunking
        generated_chunks = await self._load_chunks(request.document_ids)
        logger.info(f"✅ Fallback chunking generated {len(generated_chunks)} chunks")
        return generated_chunks

    async def _deduplicate_chunks(
        self,
        chunks: List[Chunk],
        enabled: bool
    ) -> Tuple[List[Chunk], DeduplicationStats]:
        """Deduplicate chunks using SHA256 hashing.

        Args:
            chunks: List of chunks to deduplicate
            enabled: Whether deduplication is enabled

        Returns:
            Tuple of (unique_chunks, deduplication_statistics)

        Raises:
            DeduplicationError: If deduplication fails
        """
        if not enabled:
            logger.info("Deduplication disabled, returning all chunks")
            return chunks, DeduplicationStats(
                total_chunks=len(chunks),
                unique_chunks=len(chunks),
                duplicates_removed=0,
                duplicates_percentage=0.0
            )

        if not chunks:
            return [], DeduplicationStats(0, 0, 0, 0.0)

        try:
            # Generate content hashes
            content_hashes = {}
            chunks_by_hash = {}

            for chunk in chunks:
                # Generate SHA256 hash of chunk content
                content_hash = hashlib.sha256(chunk.text.encode('utf-8')).hexdigest()

                if content_hash not in content_hashes:
                    content_hashes[content_hash] = chunk
                    chunks_by_hash[content_hash] = [chunk]
                else:
                    chunks_by_hash[content_hash].append(chunk)

            # Select best chunk for each hash (prioritize chunks with more metadata)
            unique_chunks = []
            for content_hash, chunk_list in chunks_by_hash.items():
                if len(chunk_list) == 1:
                    unique_chunks.append(chunk_list[0])
                else:
                    # Select chunk with richest metadata
                    best_chunk = max(chunk_list, key=lambda c: len(c.metadata.__dict__ if c.metadata else {}))
                    unique_chunks.append(best_chunk)

            duplicates_removed = len(chunks) - len(unique_chunks)
            duplicates_percentage = (duplicates_removed / len(chunks)) * 100 if chunks else 0

            dedup_stats = DeduplicationStats(
                total_chunks=len(chunks),
                unique_chunks=len(unique_chunks),
                duplicates_removed=duplicates_removed,
                duplicates_percentage=duplicates_percentage
            )

            logger.info(f"Deduplication complete: {len(unique_chunks)} unique chunks from {len(chunks)} total "
                       f"({duplicates_removed} duplicates removed)")

            return unique_chunks, dedup_stats

        except Exception as e:
            raise DeduplicationError(
                f"Failed to deduplicate chunks: {str(e)}",
                chunks_processed=len(chunks),
                duplicates_found=0
            )

    async def _generate_embeddings(
        self,
        chunks: List[Chunk],
        request: BuildIngestionPipelineRequest
    ) -> EmbedChunksResponse:
        """Generate embeddings for chunks using existing embedding use case.

        Args:
            chunks: Chunks to embed
            request: Pipeline request with configuration

        Returns:
            Embedding generation results

        Raises:
            EmbeddingGenerationError: If embedding generation fails
        """
        if not chunks:
            logger.warning("No chunks provided for embedding generation")
            return EmbedChunksResponse(
                success=True,
                total_chunks=0,
                embedded_count=0,
                stored_count=0
            )

        embedding_request = EmbedChunksRequest(
            chunks=chunks,
            model_name="Alibaba-NLP/gte-multilingual-base",
            model_version="1.0",
            batch_size=request.batch_size,
            show_progress=request.show_progress,
            store_embeddings=True
        )

        return await self._embedding_use_case.execute(embedding_request)

    async def _retry_operation(
        self,
        operation,
        max_retries: int,
        backoff_factor: float,
        *args,
        **kwargs
    ) -> Any:
        """Retry operation with exponential backoff.

        Args:
            operation: Async operation to retry
            max_retries: Maximum number of retry attempts
            backoff_factor: Base backoff factor in seconds
            *args: Arguments to pass to operation
            **kwargs: Keyword arguments to pass to operation

        Returns:
            Result of operation

        Raises:
            Original exception if all retries are exhausted
        """
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return await operation(*args, **kwargs)
            except (EmbeddingGenerationError, EmbeddingStorageError) as e:
                last_exception = e

                if attempt == max_retries:
                    logger.error(f"Operation failed after {max_retries + 1} attempts: {str(e)}")
                    raise

                wait_time = backoff_factor * (2 ** attempt)
                logger.warning(f"Operation failed (attempt {attempt + 1}), retrying in {wait_time}s: {str(e)}")
                await asyncio.sleep(wait_time)
                continue
            except Exception as e:
                # Don't retry on unexpected exceptions
                logger.error(f"Unexpected error in operation: {str(e)}")
                raise

        # This should not be reached, but just in case
        if last_exception:
            raise last_exception

    def get_current_job(self) -> Optional[ProcessingJob]:
        """Get the current processing job.

        Returns:
            Current processing job if one exists
        """
        return self._current_job