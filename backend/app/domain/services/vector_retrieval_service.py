"""Vector retrieval domain service.

This module contains the VectorRetrievalService domain service that orchestrates
vector-based retrieval operations with business logic and advanced features.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from app.domain.entities.query_context import QueryContext
from app.domain.entities.retrieval_result import RetrievalResult
from app.domain.entities.vector_retrieval_config import VectorRetrievalConfig
from app.domain.exceptions.retrieval_exceptions import (
    QueryValidationError,
    RetrievalError,
    SearchTimeoutError,
)
from app.domain.repositories.vector_repository import VectorRepository
from app.domain.services.embedding_service import EmbeddingService
from app.domain.services.reranking_service import RerankingService


class VectorRetrievalService:
    """Domain service for vector-based retrieval operations.

    This service orchestrates the complete vector retrieval workflow including
    query validation, embedding generation, similarity search, reranking, and
    result processing with business logic.
    """

    def __init__(
        self,
        vector_repository: VectorRepository,
        embedding_service: EmbeddingService,
        reranking_service: Optional[RerankingService] = None,
    ) -> None:
        """Initialize vector retrieval service.

        Args:
            vector_repository: Repository for vector storage and retrieval
            embedding_service: Service for generating query embeddings
            reranking_service: Optional service for result reranking
        """
        self._vector_repository = vector_repository
        self._embedding_service = embedding_service
        self._reranking_service = reranking_service

    async def retrieve_similar_documents(
        self,
        query: str,
        config: VectorRetrievalConfig,
        context: Optional[QueryContext] = None,
    ) -> RetrievalResult:
        """Execute vector-based document retrieval with complete business logic.

        Args:
            query: Search query string
            config: Retrieval configuration
            context: Optional query context for additional metadata

        Returns:
            RetrievalResult with documents and metadata

        Raises:
            QueryValidationError: If query or configuration is invalid
            RetrievalError: If retrieval operation fails
            SearchTimeoutError: If search operation times out
        """
        start_time = datetime.utcnow()
        query_id = context.query_id if context else UUID()

        try:
            # Step 1: Validate input
            self._validate_retrieval_request(query, config)

            # Step 2: Generate query embedding
            query_embedding = await self._generate_query_embedding(query, config)

            # Step 3: Apply filters from context if available
            metadata_filters = self._build_metadata_filters(config, context)

            # Step 4: Execute similarity search
            search_start = datetime.utcnow()
            search_results = await self._vector_repository.similarity_search(
                embedding=query_embedding,
                threshold=config.similarity_threshold,
                limit=config.max_results,
                filters=metadata_filters,
                namespace=config.namespace,
            )
            search_time = (datetime.utcnow() - search_start).total_seconds() * 1000

            # Step 5: Apply reranking if enabled
            if config.should_apply_reranking() and self._reranking_service:
                rerank_start = datetime.utcnow()
                search_results = await self._reranking_service.rerank(
                    query=query,
                    results=search_results,
                    top_k=config.rerank_top_k,
                    strategy=config.reranking_strategy,
                )
                rerank_time = (datetime.utcnow() - rerank_start).total_seconds() * 1000
            else:
                rerank_time = 0.0

            # Step 6: Convert to domain objects
            documents = [
                self._convert_search_result_to_document(result, idx + 1)
                for idx, result in enumerate(search_results)
            ]
            similarity_scores = [result.similarity_score.value for result in search_results]

            # Step 7: Create retrieval result
            retrieval_result = RetrievalResult(
                documents=documents,
                similarity_scores=similarity_scores,
                query_id=str(query_id),
                config=config,
                total_searched=len(search_results),
                search_time_ms=search_time,
                processing_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                metadata={
                    "query_length": len(query),
                    "embedding_dimension": query_embedding.dimension,
                    "reranking_applied": config.should_apply_reranking(),
                    "rerank_time_ms": rerank_time,
                    "model_name": query_embedding.model_name,
                    "context": context.to_dict() if context else {},
                },
                timestamp=start_time,
                reranking_applied=config.should_apply_reranking(),
            )

            # Step 8: Apply business rules
            final_result = retrieval_result.apply_business_rules()

            return final_result

        except QueryValidationError:
            raise
        except RetrievalError:
            raise
        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            raise RetrievalError(
                f"Vector retrieval failed: {str(e)}",
                operation="vector_retrieval",
                query_id=query_id,
                details=f"Processing time: {processing_time:.2f}ms"
            ) from e

    async def execute_hybrid_search(
        self,
        vector_query: str,
        keyword_query: Optional[str] = None,
        config: VectorRetrievalConfig,
        context: Optional[QueryContext] = None,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> RetrievalResult:
        """Execute hybrid search combining vector and keyword search.

        Args:
            vector_query: Vector-based search query
            keyword_query: Optional keyword-based search query
            config: Retrieval configuration
            context: Optional query context
            vector_weight: Weight for vector search results (0.0-1.0)
            keyword_weight: Weight for keyword search results (0.0-1.0)

        Returns:
            RetrievalResult with combined and ranked documents
        """
        if not config.search_mode.supports_hybrid_search():
            raise QueryValidationError(
                f"Search mode {config.search_mode.value} does not support hybrid search"
            )

        if abs(vector_weight + keyword_weight - 1.0) > 0.01:
            raise QueryValidationError(
                f"Weights must sum to 1.0: vector({vector_weight}) + keyword({keyword_weight}) = {vector_weight + keyword_weight}"
            )

        start_time = datetime.utcnow()
        query_id = context.query_id if context else UUID()

        try:
            # Execute vector search
            vector_result = await self.retrieve_similar_documents(
                query=vector_query,
                config=config,
                context=context,
            )

            # Execute keyword search if keyword query is provided
            keyword_results = []
            if keyword_query and self._vector_repository.supports_keyword_search():
                keyword_results = await self._vector_repository.keyword_search(
                    query=keyword_query,
                    limit=config.max_results,
                    filters=self._build_metadata_filters(config, context),
                )

            # Combine results using weighted scoring
            combined_results = self._combine_hybrid_results(
                vector_results=vector_result,
                keyword_results=keyword_results,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
            )

            # Create final result
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            return RetrievalResult(
                documents=[doc for doc, _ in combined_results],
                similarity_scores=[score for _, score in combined_results],
                query_id=str(query_id),
                config=config,
                total_searched=len(combined_results),
                search_time_ms=vector_result.search_time_ms,
                processing_time_ms=processing_time,
                metadata={
                    "search_type": "hybrid",
                    "vector_weight": vector_weight,
                    "keyword_weight": keyword_weight,
                    "vector_results_count": len(vector_result.documents),
                    "keyword_results_count": len(keyword_results),
                    "context": context.to_dict() if context else {},
                },
                timestamp=start_time,
                reranking_applied=False,
            )

        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            raise RetrievalError(
                f"Hybrid search failed: {str(e)}",
                operation="hybrid_search",
                query_id=query_id,
                details=f"Processing time: {processing_time:.2f}ms"
            ) from e

    async def batch_retrieve(
        self,
        queries: List[str],
        config: VectorRetrievalConfig,
        context: Optional[QueryContext] = None,
    ) -> List[RetrievalResult]:
        """Execute batch retrieval for multiple queries.

        Args:
            queries: List of search queries
            config: Retrieval configuration
            context: Optional query context

        Returns:
            List of retrieval results, one for each query
        """
        if not queries:
            return []

        if len(queries) > 100:  # Reasonable batch size limit
            raise QueryValidationError("Batch size too large: maximum 100 queries per batch")

        results = []
        for query in queries:
            try:
                result = await self.retrieve_similar_documents(query, config, context)
                results.append(result)
            except Exception as e:
                # Create error result for failed query
                error_result = RetrievalResult(
                    documents=[],
                    similarity_scores=[],
                    query_id=str(UUID()),
                    config=config,
                    total_searched=0,
                    search_time_ms=0.0,
                    processing_time_ms=0.0,
                    metadata={"error": str(e), "query": query},
                    timestamp=datetime.utcnow(),
                    reranking_applied=False,
                )
                results.append(error_result)

        return results

    def _validate_retrieval_request(self, query: str, config: VectorRetrievalConfig) -> None:
        """Validate retrieval request parameters."""
        if not query or not query.strip():
            raise QueryValidationError("Query cannot be empty")

        if len(query) > 10000:  # Reasonable query length limit
            raise QueryValidationError("Query too long: maximum 10000 characters")

        # Validate configuration (already done in __post_init__)
        # Additional business validation
        if config.is_strict_search() and len(query) < 3:
            raise QueryValidationError("Strict search requires queries of at least 3 characters")

    async def _generate_query_embedding(self, query: str, config: VectorRetrievalConfig) -> Any:
        """Generate embedding for the query."""
        try:
            return await self._embedding_service.generate_embedding(
                text=query.strip(),
                model_name=config.metadata.get("embedding_model"),
            )
        except Exception as e:
            raise RetrievalError(
                f"Failed to generate query embedding: {str(e)}",
                operation="embedding_generation"
            ) from e

    def _build_metadata_filters(
        self,
        config: VectorRetrievalConfig,
        context: Optional[QueryContext] = None,
    ) -> Dict[str, Any]:
        """Build metadata filters from config and context."""
        filters = {}

        # Add config filters
        if config.metadata_filters:
            filters.update(config.metadata_filters)

        # Add context-based filters
        if context:
            context_filters = context.get_metadata_filters()
            if context_filters:
                filters.update(context_filters)

            # Add user-specific filters
            if context.has_user_context:
                user_filters = context.get_user_preference("metadata_filters", {})
                if user_filters:
                    filters.update(user_filters)

        return filters

    def _convert_search_result_to_document(self, search_result: Any, rank: int) -> Any:
        """Convert search result to document reference."""
        # This would convert from the repository's search result format
        # to the domain's DocumentReference format
        # Implementation depends on the specific repository interface

        # Placeholder implementation - would need actual conversion logic
        from app.domain.entities.document_reference import DocumentReference

        return DocumentReference(
            id=getattr(search_result, 'id', str(UUID())),
            content=getattr(search_result, 'content', ''),
            metadata=getattr(search_result, 'metadata', {}),
            source_uri=getattr(search_result, 'source_uri', None),
            chunk_index=getattr(search_result, 'chunk_index', 0),
        )

    def _combine_hybrid_results(
        self,
        vector_results: RetrievalResult,
        keyword_results: List[Any],
        vector_weight: float,
        keyword_weight: float,
    ) -> List[tuple[Any, float]]:
        """Combine vector and keyword search results."""
        combined = {}

        # Add vector results with weighted scores
        for doc, score in zip(vector_results.documents, vector_results.similarity_scores):
            doc_id = getattr(doc, 'id', str(doc))
            combined[doc_id] = {
                'document': doc,
                'vector_score': score,
                'keyword_score': 0.0,
            }

        # Add keyword results with weighted scores
        for result in keyword_results:
            doc_id = getattr(result, 'id', str(result))
            if doc_id in combined:
                combined[doc_id]['keyword_score'] = result.score
            else:
                # Convert keyword result to document if needed
                doc = self._convert_search_result_to_document(result, len(combined) + 1)
                combined[doc_id] = {
                    'document': doc,
                    'vector_score': 0.0,
                    'keyword_score': result.score,
                }

        # Calculate final weighted scores
        final_results = []
        for item in combined.values():
            final_score = (
                item['vector_score'] * vector_weight +
                item['keyword_score'] * keyword_weight
            )
            final_results.append((item['document'], final_score))

        # Sort by final score (descending)
        final_results.sort(key=lambda x: x[1], reverse=True)

        return final_results

    async def get_retrieval_statistics(self, time_range_days: int = 7) -> Dict[str, Any]:
        """Get retrieval statistics for monitoring and analytics."""
        try:
            stats = await self._vector_repository.get_query_statistics(time_range_days)

            return {
                "time_range_days": time_range_days,
                "total_queries": stats.get("total_queries", 0),
                "average_response_time_ms": stats.get("average_response_time_ms", 0.0),
                "cache_hit_rate": stats.get("cache_hit_rate", 0.0),
                "average_similarity_score": stats.get("average_similarity_score", 0.0),
                "top_search_terms": stats.get("top_search_terms", []),
                "error_rate": stats.get("error_rate", 0.0),
                "service_status": "operational",
            }
        except Exception as e:
            return {
                "time_range_days": time_range_days,
                "error": str(e),
                "service_status": "error",
            }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of the retrieval service."""
        try:
            # Check vector repository health
            repo_health = await self._vector_repository.health_check()

            # Check embedding service health
            embedding_health = await self._embedding_service.health_check()

            # Check reranking service health if available
            reranking_health = {"status": "not_configured"}
            if self._reranking_service:
                reranking_health = await self._reranking_service.health_check()

            overall_status = "healthy"
            if any(
                health.get("status") != "healthy"
                for health in [repo_health, embedding_health, reranking_health]
            ):
                overall_status = "degraded"

            return {
                "status": overall_status,
                "vector_repository": repo_health,
                "embedding_service": embedding_health,
                "reranking_service": reranking_health,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }