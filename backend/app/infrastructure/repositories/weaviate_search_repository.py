"""Weaviate search repository implementation (Task 2.4.1).

This module implements the SearchRepository interface using Weaviate
vector database with support for semantic similarity search, filtering,
and advanced retrieval operations.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from app.domain.entities.search_query import SearchQuery
from app.domain.entities.search_result import SearchResult
from app.domain.exceptions.retrieval_exceptions import (
    FilterError,
    IndexNotFoundError,
    RetrievalError,
    SearchExecutionError,
    SearchTimeoutError,
)
from app.domain.repositories.search_repository import SearchRepository
from app.domain.repositories.embedding_repository import EmbeddingRepository
from app.domain.value_objects.similarity_score import SimilarityScore, SimilarityMethod
from app.infrastructure.repositories.weaviate_embedding_repository import (
    WeaviateEmbeddingRepository,
)
from app.infrastructure.external_services.vector_stores.weaviate_client import (
    WeaviateClient,
)

logger = logging.getLogger(__name__)


class WeaviateSearchRepository(SearchRepository):
    """Weaviate-based search repository implementation (Task 2.4.1).

    Implements semantic search using Weaviate vector database with support
    for filtering, ranking, and advanced retrieval operations.

    This repository wraps the existing WeaviateEmbeddingRepository and
    extends it with search-specific functionality while leveraging
    the existing similarity search capabilities.

    Examples:
        >>> client = WeaviateClient(config)
        >>> await client.connect()
        >>> embedding_repo = WeaviateEmbeddingRepository(client)
        >>> search_repo = WeaviateSearchRepository(client, embedding_repo)
        >>> results = await search_repo.semantic_search(search_query)
    """

    def __init__(
        self,
        weaviate_client: WeaviateClient,
        embedding_repository: EmbeddingRepository,
        embedding_service,
    ) -> None:
        """Initialize Weaviate search repository.

        Args:
            weaviate_client: Weaviate client instance
            embedding_repository: Existing embedding repository for reuse
            embedding_service: Embedding service for query processing
        """
        self._client = weaviate_client
        self._embedding_repository = embedding_repository
        self._embedding_service = embedding_service

        # Ensure embedding repository is WeaviateEmbeddingRepository
        if not isinstance(embedding_repository, WeaviateEmbeddingRepository):
            raise ValueError("embedding_repository must be a WeaviateEmbeddingRepository")

    @property
    def embedding_service(self):
        """Get the embedding service."""
        return self._embedding_service

    @property
    def embedding_repository(self):
        """Get the embedding repository."""
        return self._embedding_repository

    async def semantic_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform semantic similarity search.

        Args:
            query: Search query with text, filters, and configuration

        Returns:
            List of search results ranked by relevance

        Raises:
            RetrievalError: If search operation fails
        """
        try:
            logger.info(f"Performing semantic search for query: {query.text[:100]}...")

            # Convert SearchFilters to Weaviate filters
            weaviate_filters = None
            if not query.filters.is_empty():
                try:
                    weaviate_filters = query.filters.to_weaviate_filters()
                except Exception as e:
                    raise FilterError(
                        message=f"Failed to convert search filters: {str(e)}",
                        query_id=query.query_id,
                        filter_expression=str(query.filters),
                    ) from e

            # Embed the query using injected embedding service
            query_vector = await self._embedding_service.embed_query(
                query=query.text,
                use_cache=True,
            )

            # Use the embedded query vector for similarity search
            if query_vector is None:
                raise SearchExecutionError(
                    message="Failed to generate query embedding",
                    query_id=query.query_id,
                )

            # Use the embedding repository's similarity search
            embedding_results = await self._embedding_repository.similarity_search(
                query_vector=query_vector,
                k=query.top_k,
                filters=weaviate_filters,
                min_similarity=query.min_similarity,
            )

            # Convert embedding results to search results
            search_results = []
            for rank, (embedding, similarity) in enumerate(embedding_results, start=1):
                # Convert similarity to SimilarityScore
                similarity_score = SimilarityScore.from_weaviate_certainty(
                    certainty=similarity,
                    vector_dimension=query_vector.dimension if hasattr(query_vector, 'dimension') else 768,
                ).with_rank(rank)

                # Create SearchResult
                search_result = SearchResult.create(
                    chunk_id=embedding.chunk_id or UUID(),
                    document_id=embedding.document_id or UUID(),
                    content=embedding.metadata.get("text", ""),
                    similarity_score=similarity_score,
                    rank=rank,
                    document_metadata={
                        "doc_type": embedding.metadata.get("doc_type", "unknown"),
                        "source_type": embedding.source_type,
                        "file_name": embedding.metadata.get("file_name", ""),
                        "model_name": embedding.model_name,
                        "created_at": embedding.created_at.isoformat(),
                    },
                    chunk_metadata={
                        "sequence": embedding.metadata.get("sequence", 0),
                        "token_count": embedding.metadata.get("token_count", 0),
                        "section": embedding.metadata.get("section", ""),
                    },
                )

                search_results.append(search_result)

            logger.info(f"Found {len(search_results)} results for query")
            return search_results

        except Exception as e:
            if isinstance(e, RetrievalError):
                raise
            raise SearchExecutionError(
                message=f"Semantic search failed: {str(e)}",
                query_id=query.query_id,
                details=f"query: {query.text[:50]}...",
            ) from e

    async def find_similar_chunks(
        self,
        chunk_ids: List[str],
        top_k: int = 10,
        min_similarity: float = 0.0,
    ) -> List[SearchResult]:
        """Find chunks similar to given chunk IDs.

        Args:
            chunk_ids: List of chunk IDs to find similar items for
            top_k: Maximum number of results per chunk
            min_similarity: Minimum similarity threshold

        Returns:
            List of similar chunks grouped by input chunk

        Raises:
            RetrievalError: If similarity search fails
        """
        try:
            logger.info(f"Finding similar chunks for {len(chunk_ids)} chunk IDs")

            # Get embeddings for the input chunks
            all_similar_chunks = []
            for chunk_id in chunk_ids:
                try:
                    # Find the embedding for this chunk
                    chunk_uuid = UUID(chunk_id)
                    embedding = await self._embedding_repository.find_by_chunk_id(chunk_uuid)

                    if not embedding or not embedding.vector:
                        logger.warning(f"No embedding found for chunk_id: {chunk_id}")
                        continue

                    # Search for similar chunks using this chunk's vector
                    similar_embeddings = await self._embedding_repository.similarity_search(
                        query_vector=embedding.vector,
                        k=top_k + 1,  # +1 to include the original chunk
                        min_similarity=min_similarity,
                    )

                    # Convert to SearchResults, excluding the original chunk
                    for rank, (similar_embedding, similarity) in enumerate(similar_embeddings):
                        # Skip if this is the original chunk
                        if similar_embedding.chunk_id == chunk_uuid:
                            continue

                        similarity_score = SimilarityScore.from_weaviate_certainty(
                            certainty=similarity,
                            vector_dimension=embedding.vector.dimension if hasattr(embedding.vector, 'dimension') else 768,
                        ).with_rank(rank)

                        search_result = SearchResult.create(
                            chunk_id=similar_embedding.chunk_id or UUID(),
                            document_id=similar_embedding.document_id or UUID(),
                            content=similar_embedding.metadata.get("text", ""),
                            similarity_score=similarity_score,
                            rank=rank,
                            document_metadata={
                                "doc_type": similar_embedding.metadata.get("doc_type", "unknown"),
                                "source_type": similar_embedding.source_type,
                                "file_name": similar_embedding.metadata.get("file_name", ""),
                            },
                            chunk_metadata=similar_embedding.metadata,
                        )

                        all_similar_chunks.append(search_result)

                except Exception as e:
                    logger.error(f"Error processing chunk_id {chunk_id}: {str(e)}")
                    continue

            # Remove duplicates and sort by similarity score
            unique_chunks = {}
            for chunk in all_similar_chunks:
                chunk_key = str(chunk.chunk_id)
                if (chunk_key not in unique_chunks or
                    chunk.similarity_score.score > unique_chunks[chunk_key].similarity_score.score):
                    unique_chunks[chunk_key] = chunk

            # Sort and return top results
            sorted_chunks = sorted(
                unique_chunks.values(),
                key=lambda r: r.similarity_score.score,
                reverse=True,
            )

            return sorted_chunks[:top_k]

        except Exception as e:
            raise RetrievalError(
                message=f"Failed to find similar chunks: {str(e)}",
                operation="find_similar_chunks",
                details=f"chunk_ids: {chunk_ids[:3]}...",
            ) from e

    async def search_by_document_ids(
        self,
        document_ids: List[str],
        query_text: Optional[str] = None,
        top_k: int = 50,
    ) -> List[SearchResult]:
        """Search within specific documents.

        Args:
            document_ids: List of document IDs to search within
            query_text: Optional query text for semantic filtering
            top_k: Maximum number of results per document

        Returns:
            List of chunks from specified documents

        Raises:
            RetrievalError: If document search fails
        """
        try:
            logger.info(f"Searching within {len(document_ids)} documents")

            all_results = []
            for doc_id in document_ids:
                try:
                    doc_uuid = UUID(doc_id)

                    # Find all embeddings for this document
                    embeddings = await self._embedding_repository.find_by_document_id(doc_uuid)

                    # If query text provided, rank by similarity
                    if query_text:
                        # Embed query and rank by similarity
                        from app.infrastructure.external_services.ml_services.embedding_models.sentence_transformer_service import (
                            SentenceTransformerService,
                        )
                        embedding_service = SentenceTransformerService()
                        query_vector = await embedding_service.embed_query(query_text)

                        # Calculate similarity scores
                        similar_embeddings = []
                        for embedding in embeddings:
                            if embedding.vector:
                                # Simple cosine similarity calculation
                                similarity = self._calculate_cosine_similarity(
                                    query_vector.to_list(),
                                    embedding.vector.to_list()
                                )
                                similar_embeddings.append((embedding, similarity))

                        # Sort by similarity
                        similar_embeddings.sort(key=lambda x: x[1], reverse=True)
                        embeddings = [emb for emb, _ in similar_embeddings[:top_k]]

                    # Convert to SearchResults
                    for rank, embedding in enumerate(embeddings[:top_k], start=1):
                        similarity_score = SimilarityScore.cosine_similarity(
                            score=0.8,  # Default score for document search
                            confidence=0.7,
                        ).with_rank(rank)

                        search_result = SearchResult.create(
                            chunk_id=embedding.chunk_id or UUID(),
                            document_id=embedding.document_id or UUID(),
                            content=embedding.metadata.get("text", ""),
                            similarity_score=similarity_score,
                            rank=rank,
                            document_metadata={
                                "doc_type": embedding.metadata.get("doc_type", "unknown"),
                                "source_type": embedding.source_type,
                                "file_name": embedding.metadata.get("file_name", ""),
                            },
                            chunk_metadata=embedding.metadata,
                        )

                        all_results.append(search_result)

                except Exception as e:
                    logger.error(f"Error processing document_id {doc_id}: {str(e)}")
                    continue

            # Sort by rank and similarity
            all_results.sort(key=lambda r: (r.rank, r.similarity_score.score), reverse=False)

            return all_results

        except Exception as e:
            raise RetrievalError(
                message=f"Failed to search documents: {str(e)}",
                operation="search_by_document_ids",
                details=f"document_ids: {document_ids[:3]}...",
            ) from e

    async def get_search_suggestions(
        self,
        partial_query: str,
        max_suggestions: int = 5,
        min_similarity: float = 0.3,
    ) -> List[str]:
        """Get search suggestions based on partial query.

        Args:
            partial_query: Partial search query text
            max_suggestions: Maximum number of suggestions
            min_similarity: Minimum similarity for suggestions

        Returns:
            List of suggested query completions

        Raises:
            RetrievalError: If suggestion generation fails
        """
        try:
            # For now, return simple suggestions based on common fitness terms
            # In a production system, this would use actual search analytics
            common_terms = [
                "exercises for lower back pain",
                "strength training routine",
                "cardio workout plan",
                "flexibility stretches",
                "core strengthening exercises",
                "injury rehabilitation",
                "warm up exercises",
                "cool down stretches",
                "muscle building tips",
                "weight loss workout",
            ]

            # Filter by similarity to partial query
            suggestions = []
            for term in common_terms:
                # Simple string similarity for now
                similarity = self._calculate_string_similarity(partial_query.lower(), term)
                if similarity >= min_similarity:
                    suggestions.append((term, similarity))

            # Sort by similarity and return top results
            suggestions.sort(key=lambda x: x[1], reverse=True)
            return [suggestion for suggestion, _ in suggestions[:max_suggestions]]

        except Exception as e:
            raise RetrievalError(
                message=f"Failed to get search suggestions: {str(e)}",
                operation="get_search_suggestions",
                details=f"partial_query: '{partial_query}'",
            ) from e

    async def aggregate_search_results(
        self,
        queries: List[SearchQuery],
        aggregation_method: str = "reciprocal_rank",
    ) -> List[SearchResult]:
        """Aggregate results from multiple search queries.

        Args:
            queries: List of search queries to execute
            aggregation_method: Method for combining results

        Returns:
            Aggregated list of search results

        Raises:
            RetrievalError: If aggregation fails
        """
        try:
            if not queries:
                return []

            logger.info(f"Aggregating results from {len(queries)} queries")

            # Get results for each query
            all_result_sets = []
            for query in queries:
                results = await self.semantic_search(query)
                all_result_sets.append(results)

            # Use RetrievalService for aggregation
            from app.domain.services.retrieval_service import RetrievalService
            retrieval_service = RetrievalService()

            aggregated_results = retrieval_service.aggregate_multiple_searches(
                search_results=all_result_sets,
                method=aggregation_method,
            )

            return aggregated_results

        except Exception as e:
            raise RetrievalError(
                message=f"Failed to aggregate search results: {str(e)}",
                operation="aggregate_search_results",
                details=f"queries: {len(queries)}, method: {aggregation_method}",
            ) from e

    async def get_popular_queries(
        self,
        limit: int = 10,
        time_range_days: int = 30,
    ) -> List[Tuple[str, int]]:
        """Get most popular search queries.

        Args:
            limit: Maximum number of queries to return
            time_range_days: Number of days to look back

        Returns:
            List of (query_text, frequency) tuples

        Raises:
            RetrievalError: If query analytics fail
        """
        # TODO: Implement actual search analytics database query
        # This should query a search analytics table that tracks:
        # - Query frequency and popularity
        # - User interaction patterns
        # - Performance metrics
        raise NotImplementedError(
            "Search analytics not yet implemented. "
            "Please implement a search analytics database to track query popularity."
        )

    async def log_search_interaction(
        self,
        query_id: str,
        result_ids: List[str],
        clicked_result_id: Optional[str] = None,
        feedback_score: Optional[float] = None,
    ) -> None:
        """Log user interaction with search results.

        Args:
            query_id: ID of the search query
            result_ids: IDs of returned results
            clicked_result_id: ID of result user clicked (if any)
            feedback_score: User feedback score (1-5, if provided)

        Raises:
            RetrievalError: If logging fails
        """
        try:
            # For now, just log to the application logger
            # In production, this would store to analytics database
            logger.info(
                f"Search interaction logged - query_id: {query_id}, "
                f"results: {len(result_ids)}, clicked: {clicked_result_id}, "
                f"feedback: {feedback_score}"
            )

        except Exception as e:
            raise RetrievalError(
                message=f"Failed to log search interaction: {str(e)}",
                operation="log_search_interaction",
            ) from e

    async def health_check(self) -> Dict[str, Any]:
        """Check health and performance of search repository.

        Returns:
            Dictionary with health status and metrics

        Raises:
            RetrievalError: If health check fails
        """
        try:
            # Check Weaviate connectivity
            weaviate_health = await self._embedding_repository.health_check()

            # Test basic search functionality
            test_query_text = "health check test query"
            try:
                from app.domain.value_objects.search_filters import SearchFilters
                from app.domain.entities.search_query import SearchQuery

                test_query = SearchQuery.create(
                    text=test_query_text,
                    filters=SearchFilters(),
                    top_k=1,
                )
                test_results = await self.semantic_search(test_query)
                search_working = True
            except Exception as e:
                search_working = False
                logger.warning(f"Search test failed: {str(e)}")

            return {
                "status": "healthy" if weaviate_health and search_working else "degraded",
                "weaviate_connected": weaviate_health,
                "search_functionality": search_working,
                "timestamp": datetime.utcnow().isoformat(),
                "test_query": test_query_text,
                "test_results_count": len(test_results) if search_working else 0,
            }

        except Exception as e:
            raise RetrievalError(
                message=f"Health check failed: {str(e)}",
                operation="health_check",
            ) from e

    async def get_search_statistics(self, time_range_days: int = 7) -> Dict[str, Any]:
        """Get search performance and usage statistics.

        Args:
            time_range_days: Number of days to analyze

        Returns:
            Dictionary with search statistics

        Raises:
            RetrievalError: If statistics retrieval fails
        """
        # TODO: Implement actual search analytics database query
        # This should query a search analytics table that tracks:
        # - Query volume and patterns
        # - Performance metrics
        # - Error rates and types
        # - User interaction statistics
        raise NotImplementedError(
            "Search statistics not yet implemented. "
            "Please implement a search analytics database to track performance metrics."
        )

    # Helper methods

    async def _get_query_vector(self, embedding_id: Optional[UUID]) -> Optional[Any]:
        """Get query vector from embedding ID.

        Args:
            embedding_id: ID of the embedding

        Returns:
            Query vector if found, None otherwise
        """
        if not embedding_id:
            return None

        try:
            embedding = await self._embedding_repository.find_by_id(embedding_id)
            return embedding.vector if embedding else None
        except Exception:
            return None

    def _calculate_cosine_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vector1: First vector
            vector2: Second vector

        Returns:
            Cosine similarity score (0-1)
        """
        if len(vector1) != len(vector2):
            return 0.0

        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vector1, vector2))

        # Calculate magnitudes
        mag1 = sum(a * a for a in vector1) ** 0.5
        mag2 = sum(b * b for b in vector2) ** 0.5

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot_product / (mag1 * mag2)

    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate simple string similarity.

        Args:
            str1: First string
            str2: Second string

        Returns:
            Similarity score (0-1)
        """
        # Simple Jaccard similarity for now
        set1 = set(str1.lower().split())
        set2 = set(str2.lower().split())

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0