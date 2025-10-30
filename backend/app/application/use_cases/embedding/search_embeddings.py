"""Search embeddings use case (Task 2.2.1).

This use case performs similarity search on stored embeddings to find
the most relevant chunks for a given query.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from app.domain.entities.embedding import Embedding
from app.domain.exceptions.embedding_exceptions import (
    EmbeddingGenerationError,
    EmbeddingStorageError,
)
from app.domain.repositories.embedding_repository import EmbeddingRepository
from app.domain.services.embedding_service import EmbeddingService
from app.domain.value_objects.embedding_vector import EmbeddingVector
from app.infrastructure.external_services.ml_services.embedding_models.sentence_transformer_service import (
    SentenceTransformerService,
)


@dataclass
class SearchRequest:
    """Request for similarity search.

    Supports both direct vector search and query text search.
    """

    query: Optional[str] = None
    query_vector: Optional[EmbeddingVector] = None
    k: int = 10
    min_similarity: float = 0.0
    filters: Optional[Dict[str, Any]] = None
    include_vectors: bool = False
    model_name: str = "Alibaba-NLP/gte-multilingual-base"


@dataclass
class SearchResult:
    """Single search result with similarity score."""

    embedding: Embedding
    similarity_score: float
    rank: int

    def as_dict(self, include_vector: bool = False) -> Dict[str, Any]:
        """Convert to dictionary.

        Args:
            include_vector: Whether to include vector data

        Returns:
            Dictionary representation
        """
        result = {
            "embedding_id": str(self.embedding.id),
            "chunk_id": str(self.embedding.chunk_id) if self.embedding.chunk_id else None,
            "document_id": (
                str(self.embedding.document_id) if self.embedding.document_id else None
            ),
            "similarity_score": self.similarity_score,
            "rank": self.rank,
            "metadata": self.embedding.metadata,
        }

        if include_vector and self.embedding.vector:
            result["vector_dimension"] = self.embedding.vector.dimension

        return result


@dataclass
class SearchResponse:
    """Response from similarity search."""

    success: bool
    query: Optional[str] = None
    total_results: int = 0
    processing_time_ms: float = 0.0
    results: List[SearchResult] = field(default_factory=list)
    error: Optional[str] = None

    def as_dict(self, include_vectors: bool = False) -> Dict[str, Any]:
        """Convert to dictionary.

        Args:
            include_vectors: Whether to include vector data in results

        Returns:
            Dictionary representation
        """
        return {
            "success": self.success,
            "query": self.query,
            "total_results": self.total_results,
            "processing_time_ms": self.processing_time_ms,
            "results": [r.as_dict(include_vector=include_vectors) for r in self.results],
            "error": self.error,
        }


class SearchEmbeddingsUseCase:
    """Use case for similarity search on embeddings (Task 2.2.1).

    Performs vector similarity search to find the most relevant document chunks
    for a given query. Supports both direct vector search and query text search
    with automatic embedding generation.

    Performance Target: <200ms search latency for k=10

    Examples:
        >>> use_case = SearchEmbeddingsUseCase(
        ...     embedding_service=embedding_service,
        ...     embedding_repository=repository,
        ...     domain_service=domain_service
        ... )
        >>>
        >>> # Search by query text
        >>> request = SearchRequest(
        ...     query="What are the best exercises for lower back pain?",
        ...     k=5,
        ...     min_similarity=0.7
        ... )
        >>> response = await use_case.execute(request)
        >>> response.success
        True
        >>> len(response.results) <= 5
        True
        >>> all(r.similarity_score >= 0.7 for r in response.results)
        True
        >>>
        >>> # Search by vector directly
        >>> request = SearchRequest(
        ...     query_vector=existing_vector,
        ...     k=10,
        ...     filters={"doc_type": "pdf"}
        ... )
        >>> response = await use_case.execute(request)
    """

    def __init__(
        self,
        embedding_service: SentenceTransformerService,
        embedding_repository: EmbeddingRepository,
        domain_service: EmbeddingService,
    ) -> None:
        """Initialize search embeddings use case.

        Args:
            embedding_service: Service for generating embeddings
            embedding_repository: Repository for searching embeddings
            domain_service: Domain service for coordination
        """
        self._embedding_service = embedding_service
        self._repository = embedding_repository
        self._domain_service = domain_service

    async def execute(self, request: SearchRequest) -> SearchResponse:
        """Execute similarity search.

        Args:
            request: Search request

        Returns:
            Response with search results and metrics
        """
        start_time = datetime.now()

        # Step 1: Validate request
        if not request.query and not request.query_vector:
            return SearchResponse(
                success=False,
                error="Either query or query_vector must be provided",
            )

        if request.k <= 0:
            return SearchResponse(
                success=False,
                error="k must be greater than 0",
            )

        try:
            # Step 2: Get query vector
            if request.query_vector:
                query_vector = request.query_vector
                query_text = request.query
            else:
                # Generate embedding for query text
                try:
                    query_vector = await self._embedding_service.embed_query(
                        query=request.query,
                        use_cache=True,
                    )
                    query_text = request.query
                except EmbeddingGenerationError as e:
                    processing_time = (datetime.now() - start_time).total_seconds() * 1000
                    return SearchResponse(
                        success=False,
                        query=request.query,
                        processing_time_ms=processing_time,
                        error=f"Failed to generate query embedding: {str(e)}",
                    )

            # Step 3: Perform similarity search
            try:
                search_results = await self._repository.similarity_search(
                    query_vector=query_vector,
                    k=request.k,
                    filters=request.filters,
                    min_similarity=request.min_similarity,
                )
            except EmbeddingStorageError as e:
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                return SearchResponse(
                    success=False,
                    query=query_text,
                    processing_time_ms=processing_time,
                    error=f"Search failed: {str(e)}",
                )

            # Step 4: Format results
            results = []
            for rank, (embedding, similarity_score) in enumerate(search_results, start=1):
                results.append(
                    SearchResult(
                        embedding=embedding,
                        similarity_score=similarity_score,
                        rank=rank,
                    )
                )

            # Step 5: Calculate metrics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            return SearchResponse(
                success=True,
                query=query_text,
                total_results=len(results),
                processing_time_ms=processing_time,
                results=results,
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            return SearchResponse(
                success=False,
                query=query_text if "query_text" in locals() else request.query,
                processing_time_ms=processing_time,
                error=f"Unexpected error: {str(e)}",
            )

    async def search_by_text(
        self,
        query: str,
        k: int = 10,
        min_similarity: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> SearchResponse:
        """Convenience method for text-based search.

        Args:
            query: Query text
            k: Number of results to return
            min_similarity: Minimum similarity threshold
            filters: Optional metadata filters

        Returns:
            Search response
        """
        request = SearchRequest(
            query=query,
            k=k,
            min_similarity=min_similarity,
            filters=filters,
        )
        return await self.execute(request)

    async def search_by_vector(
        self,
        query_vector: EmbeddingVector,
        k: int = 10,
        min_similarity: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> SearchResponse:
        """Convenience method for vector-based search.

        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            min_similarity: Minimum similarity threshold
            filters: Optional metadata filters

        Returns:
            Search response
        """
        request = SearchRequest(
            query_vector=query_vector,
            k=k,
            min_similarity=min_similarity,
            filters=filters,
        )
        return await self.execute(request)

    async def search_similar_to_embedding(
        self,
        embedding_id: UUID,
        k: int = 10,
        min_similarity: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> SearchResponse:
        """Search for embeddings similar to an existing embedding.

        Args:
            embedding_id: ID of reference embedding
            k: Number of results to return
            min_similarity: Minimum similarity threshold
            filters: Optional metadata filters

        Returns:
            Search response
        """
        start_time = datetime.now()

        try:
            # Retrieve reference embedding
            reference = await self._repository.find_by_id(embedding_id)

            if not reference:
                return SearchResponse(
                    success=False,
                    error=f"Embedding {embedding_id} not found",
                )

            if not reference.vector:
                return SearchResponse(
                    success=False,
                    error=f"Embedding {embedding_id} has no vector",
                )

            # Search using reference vector
            request = SearchRequest(
                query_vector=reference.vector,
                k=k + 1,  # +1 to account for reference itself
                min_similarity=min_similarity,
                filters=filters,
            )

            response = await self.execute(request)

            # Remove reference embedding from results if present
            if response.success:
                response.results = [
                    r for r in response.results if r.embedding.id != embedding_id
                ][:k]
                response.total_results = len(response.results)

            return response

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            return SearchResponse(
                success=False,
                processing_time_ms=processing_time,
                error=f"Failed to search similar embeddings: {str(e)}",
            )
