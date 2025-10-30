"""Embed document chunks use case (Task 2.2.1).

This use case generates embeddings for document chunks and stores them
in Weaviate vector database for retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID

from app.domain.entities.chunk import Chunk
from app.domain.entities.embedding import Embedding
from app.domain.exceptions.embedding_exceptions import EmbeddingGenerationError
from app.domain.repositories.embedding_repository import EmbeddingRepository
from app.domain.services.embedding_service import EmbeddingService
from app.domain.value_objects.embedding_vector import EmbeddingVector
from app.infrastructure.external_services.ml_services.embedding_models.sentence_transformer_service import (
    SentenceTransformerService,
)


@dataclass
class EmbedChunksRequest:
    """Request to embed document chunks."""

    chunks: Sequence[Chunk]
    model_name: str = "Alibaba-NLP/gte-multilingual-base"
    model_version: str = "1.0"
    batch_size: int = 32
    show_progress: bool = True
    store_embeddings: bool = True


@dataclass
class ChunkEmbeddingResult:
    """Result for single chunk embedding."""

    chunk_id: UUID
    embedding_id: Optional[UUID] = None
    success: bool = True
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_id": str(self.chunk_id),
            "embedding_id": str(self.embedding_id) if self.embedding_id else None,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class EmbedChunksResponse:
    """Response from embedding document chunks."""

    success: bool
    total_chunks: int
    embedded_count: int
    stored_count: int
    results: List[ChunkEmbeddingResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "total_chunks": self.total_chunks,
            "embedded_count": self.embedded_count,
            "stored_count": self.stored_count,
            "results": [r.as_dict() for r in self.results],
            "errors": self.errors,
        }


class EmbedDocumentChunksUseCase:
    """Use case for embedding document chunks (Task 2.2.1).

    Generates embeddings for document chunks and optionally stores them
    in the vector database for retrieval.

    Examples:
        >>> use_case = EmbedDocumentChunksUseCase(
        ...     embedding_service=embedding_service,
        ...     embedding_repository=repository,
        ...     domain_service=domain_service
        ... )
        >>> request = EmbedChunksRequest(chunks=chunks)
        >>> response = await use_case.execute(request)
        >>> response.success
        True
        >>> response.embedded_count == len(chunks)
        True
    """

    def __init__(
        self,
        embedding_service: SentenceTransformerService,
        embedding_repository: EmbeddingRepository,
        domain_service: EmbeddingService,
    ) -> None:
        """Initialize embed chunks use case.

        Args:
            embedding_service: Service for generating embeddings
            embedding_repository: Repository for storing embeddings
            domain_service: Domain service for coordination
        """
        self._embedding_service = embedding_service
        self._repository = embedding_repository
        self._domain_service = domain_service

    async def execute(self, request: EmbedChunksRequest) -> EmbedChunksResponse:
        """Execute chunk embedding.

        Args:
            request: Embed chunks request

        Returns:
            Response with embedding results
        """
        if not request.chunks:
            return EmbedChunksResponse(
                success=True,
                total_chunks=0,
                embedded_count=0,
                stored_count=0,
            )

        errors = []
        results = []

        # Step 1: Extract text from chunks
        texts = []
        valid_chunks = []
        for chunk in request.chunks:
            if chunk.text and chunk.text.strip():
                texts.append(chunk.text)
                valid_chunks.append(chunk)
            else:
                results.append(
                    ChunkEmbeddingResult(
                        chunk_id=chunk.chunk_id,
                        success=False,
                        error="Empty chunk text",
                    )
                )

        if not valid_chunks:
            return EmbedChunksResponse(
                success=False,
                total_chunks=len(request.chunks),
                embedded_count=0,
                stored_count=0,
                results=results,
                errors=["No valid chunks to embed"],
            )

        # Step 2: Generate embeddings in batch
        try:
            vectors = await self._embedding_service.embed_batch(
                texts=texts,
                batch_size=request.batch_size,
                show_progress=request.show_progress,
            )
        except EmbeddingGenerationError as e:
            errors.append(f"Failed to generate embeddings: {str(e)}")
            return EmbedChunksResponse(
                success=False,
                total_chunks=len(request.chunks),
                embedded_count=0,
                stored_count=0,
                results=results,
                errors=errors,
            )

        # Step 3: Store embeddings if requested
        stored_count = 0
        if request.store_embeddings:
            try:
                stored_embeddings = await self._domain_service.store_chunk_embeddings_batch(
                    chunks=valid_chunks,
                    vectors=vectors,
                    model_name=request.model_name,
                    model_version=request.model_version,
                    batch_size=request.batch_size,
                )
                stored_count = len(stored_embeddings)

                # Create success results
                for chunk, embedding in zip(valid_chunks, stored_embeddings):
                    results.append(
                        ChunkEmbeddingResult(
                            chunk_id=chunk.chunk_id,
                            embedding_id=embedding.id,
                            success=True,
                        )
                    )

            except Exception as e:
                errors.append(f"Failed to store embeddings: {str(e)}")
                # Create results without storage
                for chunk in valid_chunks:
                    results.append(
                        ChunkEmbeddingResult(
                            chunk_id=chunk.chunk_id,
                            success=True,
                            error="Embedding generated but not stored",
                        )
                    )
        else:
            # Just mark as embedded
            for chunk in valid_chunks:
                results.append(
                    ChunkEmbeddingResult(
                        chunk_id=chunk.chunk_id,
                        success=True,
                    )
                )

        success = len(errors) == 0

        return EmbedChunksResponse(
            success=success,
            total_chunks=len(request.chunks),
            embedded_count=len(vectors),
            stored_count=stored_count,
            results=results,
            errors=errors,
        )
