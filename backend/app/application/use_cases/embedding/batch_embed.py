"""Batch embedding use case (Task 2.2.1).

This use case handles large-scale embedding operations with progress tracking,
flexible batching, and performance optimization for high-throughput scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID, uuid4

from app.domain.entities.embedding import Embedding
from app.domain.exceptions.embedding_exceptions import EmbeddingGenerationError
from app.domain.repositories.embedding_repository import EmbeddingRepository
from app.domain.services.embedding_service import EmbeddingService
from app.domain.value_objects.embedding_vector import EmbeddingVector
from app.infrastructure.external_services.ml_services.embedding_models.sentence_transformer_service import (
    SentenceTransformerService,
)


@dataclass
class TextItem:
    """Individual text item for batch embedding."""

    text: str
    item_id: UUID = field(default_factory=uuid4)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchEmbedRequest:
    """Request for batch embedding operation.

    Supports both simple text lists and structured text items with metadata.
    """

    texts: Optional[Sequence[str]] = None
    text_items: Optional[Sequence[TextItem]] = None
    model_name: str = "Alibaba-NLP/gte-multilingual-base"
    model_version: str = "1.0"
    batch_size: int = 32
    show_progress: bool = True
    store_embeddings: bool = False
    storage_batch_size: int = 100


@dataclass
class BatchEmbedResult:
    """Result for single batch embedding item."""

    item_id: UUID
    embedding_id: Optional[UUID] = None
    success: bool = True
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "item_id": str(self.item_id),
            "embedding_id": str(self.embedding_id) if self.embedding_id else None,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class BatchEmbedResponse:
    """Response from batch embedding operation."""

    success: bool
    total_items: int
    embedded_count: int
    stored_count: int
    failed_count: int
    processing_time_ms: float
    throughput_items_per_second: float
    results: List[BatchEmbedResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "total_items": self.total_items,
            "embedded_count": self.embedded_count,
            "stored_count": self.stored_count,
            "failed_count": self.failed_count,
            "processing_time_ms": self.processing_time_ms,
            "throughput_items_per_second": self.throughput_items_per_second,
            "results": [r.as_dict() for r in self.results],
            "errors": self.errors,
        }


class BatchEmbedUseCase:
    """Use case for batch embedding operations (Task 2.2.1).

    Handles large-scale embedding generation with:
    - Flexible batch processing with configurable batch sizes
    - Progress tracking for long-running operations
    - Performance metrics and throughput monitoring
    - Optional storage with batch optimization
    - Comprehensive error handling and recovery

    Performance Target: ≥1000 texts/minute (16.7 texts/second)

    Examples:
        >>> use_case = BatchEmbedUseCase(
        ...     embedding_service=embedding_service,
        ...     embedding_repository=repository,
        ...     domain_service=domain_service
        ... )
        >>>
        >>> # Simple text list
        >>> request = BatchEmbedRequest(
        ...     texts=["Exercise 1", "Exercise 2", "Exercise 3"],
        ...     batch_size=32
        ... )
        >>> response = await use_case.execute(request)
        >>> response.success
        True
        >>> response.embedded_count == 3
        True
        >>>
        >>> # Structured text items with metadata
        >>> items = [
        ...     TextItem(text="Text 1", metadata={"category": "A"}),
        ...     TextItem(text="Text 2", metadata={"category": "B"})
        ... ]
        >>> request = BatchEmbedRequest(text_items=items)
        >>> response = await use_case.execute(request)
    """

    def __init__(
        self,
        embedding_service: SentenceTransformerService,
        embedding_repository: EmbeddingRepository,
        domain_service: EmbeddingService,
    ) -> None:
        """Initialize batch embed use case.

        Args:
            embedding_service: Service for generating embeddings
            embedding_repository: Repository for storing embeddings
            domain_service: Domain service for coordination
        """
        self._embedding_service = embedding_service
        self._repository = embedding_repository
        self._domain_service = domain_service

    async def execute(self, request: BatchEmbedRequest) -> BatchEmbedResponse:
        """Execute batch embedding operation.

        Args:
            request: Batch embed request

        Returns:
            Response with batch results and performance metrics
        """
        start_time = datetime.now()

        # Step 1: Validate and prepare input
        text_items = self._prepare_text_items(request)

        if not text_items:
            return BatchEmbedResponse(
                success=False,
                total_items=0,
                embedded_count=0,
                stored_count=0,
                failed_count=0,
                processing_time_ms=0.0,
                throughput_items_per_second=0.0,
                errors=["No valid text items provided"],
            )

        errors = []
        results = []

        # Step 2: Extract and validate texts
        valid_items = []
        for item in text_items:
            if item.text and item.text.strip():
                valid_items.append(item)
            else:
                results.append(
                    BatchEmbedResult(
                        item_id=item.item_id,
                        success=False,
                        error="Empty text",
                    )
                )

        if not valid_items:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            return BatchEmbedResponse(
                success=False,
                total_items=len(text_items),
                embedded_count=0,
                stored_count=0,
                failed_count=len(text_items),
                processing_time_ms=processing_time,
                throughput_items_per_second=0.0,
                results=results,
                errors=["No valid texts to embed"],
            )

        # Step 3: Generate embeddings in batches
        texts = [item.text for item in valid_items]

        try:
            vectors = await self._embedding_service.embed_batch(
                texts=texts,
                batch_size=request.batch_size,
                show_progress=request.show_progress,
            )
        except EmbeddingGenerationError as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            errors.append(f"Failed to generate embeddings: {str(e)}")
            return BatchEmbedResponse(
                success=False,
                total_items=len(text_items),
                embedded_count=0,
                stored_count=0,
                failed_count=len(text_items),
                processing_time_ms=processing_time,
                throughput_items_per_second=0.0,
                results=results,
                errors=errors,
            )

        # Step 4: Optionally store embeddings
        stored_count = 0
        if request.store_embeddings:
            try:
                # Create embedding entities
                embeddings = []
                for item, vector in zip(valid_items, vectors):
                    metadata = {
                        "text": item.text,
                        "source_type": "batch",
                        **item.metadata,
                    }

                    embedding = Embedding(
                        vector=vector,
                        model_name=request.model_name,
                        model_version=request.model_version,
                        metadata=metadata,
                    )
                    embeddings.append(embedding)

                # Batch save with optimization
                stored_count = await self._repository.batch_save(
                    embeddings,
                    batch_size=request.storage_batch_size,
                )

                # Create success results with embedding IDs
                for item, embedding in zip(valid_items, embeddings):
                    results.append(
                        BatchEmbedResult(
                            item_id=item.item_id,
                            embedding_id=embedding.id,
                            success=True,
                        )
                    )

            except Exception as e:
                errors.append(f"Failed to store embeddings: {str(e)}")
                # Create results without storage
                for item in valid_items:
                    results.append(
                        BatchEmbedResult(
                            item_id=item.item_id,
                            success=True,
                            error="Embedding generated but not stored",
                        )
                    )
        else:
            # Just mark as embedded
            for item in valid_items:
                results.append(
                    BatchEmbedResult(
                        item_id=item.item_id,
                        success=True,
                    )
                )

        # Step 5: Calculate metrics
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        throughput = (
            len(vectors) / (processing_time / 1000) if processing_time > 0 else 0.0
        )

        success = len(errors) == 0
        failed_count = len(text_items) - len(vectors)

        return BatchEmbedResponse(
            success=success,
            total_items=len(text_items),
            embedded_count=len(vectors),
            stored_count=stored_count,
            failed_count=failed_count,
            processing_time_ms=processing_time,
            throughput_items_per_second=throughput,
            results=results,
            errors=errors,
        )

    def _prepare_text_items(self, request: BatchEmbedRequest) -> List[TextItem]:
        """Prepare text items from request.

        Args:
            request: Batch embed request

        Returns:
            List of text items
        """
        if request.text_items:
            return list(request.text_items)
        elif request.texts:
            return [TextItem(text=text) for text in request.texts]
        else:
            return []
