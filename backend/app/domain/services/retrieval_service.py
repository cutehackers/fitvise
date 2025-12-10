"""Retrieval service for coordinating search operations (Task 2.4.1).

This module defines the RetrievalService domain service that coordinates
search result processing, ranking, and formatting for semantic search.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.domain.entities.search_query import SearchQuery
from app.domain.entities.search_result import SearchResult
from app.domain.repositories.search_repository import SearchRepository
from app.domain.services.embedding_service import EmbeddingService
from app.domain.value_objects.embedding_result import EmbeddingResult
from app.domain.value_objects.similarity_score import SimilarityScore
from datetime import datetime, timezone


class RetrievalService:
    """Domain service for coordinating retrieval operations.

    Provides business logic for processing search results, applying ranking
    algorithms, formatting results for presentation, and orchestrating
    complete semantic search workflows.

    Examples:
        >>> service = RetrievalService(embedding_service, search_repository)
        >>> # Execute complete semantic search
        >>> results = await service.semantic_search(
        ...     query="What exercises help with back pain?",
        ...     top_k=5,
        ...     filters={"doc_type": "fitness"}
        ... )
        >>> len(results)
        5
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        search_repository: SearchRepository,
    ) -> None:
        """Initialize retrieval service.

        Args:
            embedding_service: Service for generating embeddings
            search_repository: Repository for search operations
        """
        self._embedding_service = embedding_service
        self._search_repository = search_repository

    async def process_search_results(
        self,
        raw_results: List[SearchResult],
        query: SearchQuery,
        query_embedding_dimension: int = 0,
    ) -> List[SearchResult]:
        """Process and format raw search results.

        Args:
            raw_results: Raw search results from repository
            query: Original search query
            query_embedding_dimension: Dimension of query embedding vector

        Returns:
            Processed and ranked search results
        """
        if not raw_results:
            return []

        # Step 1: Filter results by minimum similarity threshold
        filtered_results = [
            result for result in raw_results
            if result.is_above_threshold(query.min_similarity)
        ]

        # Step 2: Apply additional ranking if needed
        ranked_results = self._apply_ranking_algorithm(
            results=filtered_results,
            query=query,
        )

        # Step 3: Limit to top_k results
        limited_results = ranked_results[:query.top_k]

        # Step 4: Add context and highlights if metadata is requested
        if query.include_metadata:
            limited_results = [
                self._enhance_result_metadata(result, query)
                for result in limited_results
            ]

        return limited_results

    def _apply_ranking_algorithm(
        self,
        results: List[SearchResult],
        query: SearchQuery,
    ) -> List[SearchResult]:
        """Apply ranking algorithm to search results.

        Args:
            results: Search results to rank
            query: Original search query for context

        Returns:
            Re-ranked search results
        """
        if not results:
            return []

        # Sort by similarity score (descending) and original rank
        sorted_results = sorted(
            results,
            key=lambda r: (r.similarity_score.score, -r.rank),
            reverse=True,
        )

        # Update ranks
        ranked_results = []
        for idx, result in enumerate(sorted_results):
            # Create new result with updated rank
            updated_score = result.similarity_score.with_rank(idx + 1)
            ranked_results.append(SearchResult(
                result_id=result.result_id,
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                content=result.content,
                similarity_score=updated_score,
                rank=idx + 1,
                document_metadata=result.document_metadata,
                chunk_metadata=result.chunk_metadata,
                highlight_text=result.highlight_text,
                context_before=result.context_before,
                context_after=result.context_after,
                created_at=result.created_at,
                metadata=result.metadata,
            ))

        return ranked_results

    def _enhance_result_metadata(
        self,
        result: SearchResult,
        query: SearchQuery,
    ) -> SearchResult:
        """Enhance result with additional metadata and context.

        Args:
            result: Search result to enhance
            query: Original search query for context

        Returns:
            Enhanced search result
        """
        # Add quality score to metadata
        enhanced_metadata = result.metadata.copy()
        enhanced_metadata.update({
            "search_quality": self._calculate_search_quality(result),
            "content_length": len(result.content),
            "matches_query_terms": self._check_query_term_matches(result.content, query.text),
        })

        # Generate highlight if not present
        highlight_text = result.highlight_text
        if not highlight_text:
            highlight_text = self._generate_highlight(result.content, query.text)

        return SearchResult(
            result_id=result.result_id,
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            content=result.content,
            similarity_score=result.similarity_score,
            rank=result.rank,
            document_metadata=result.document_metadata,
            chunk_metadata=result.chunk_metadata,
            highlight_text=highlight_text,
            context_before=result.context_before,
            context_after=result.context_after,
            created_at=result.created_at,
            metadata=enhanced_metadata,
        )

    def _calculate_search_quality(self, result: SearchResult) -> str:
        """Calculate overall search quality score.

        Args:
            result: Search result to evaluate

        Returns:
            Quality rating string
        """
        similarity = result.similarity_score.score
        confidence = result.similarity_score.confidence
        content_length = len(result.content)

        # Quality scoring logic
        if similarity >= 0.9 and confidence >= 0.8:
            return "excellent"
        elif similarity >= 0.8 and confidence >= 0.7:
            return "very_good"
        elif similarity >= 0.7 and confidence >= 0.6:
            return "good"
        elif similarity >= 0.6:
            return "fair"
        elif content_length > 100:  # Longer content might be useful even with lower similarity
            return "potentially_useful"
        else:
            return "poor"

    def _check_query_term_matches(self, content: str, query: str) -> List[str]:
        """Check which query terms appear in the content.

        Args:
            content: Result content
            query: Original query text

        Returns:
            List of matching query terms
        """
        if not content or not query:
            return []

        # Simple keyword matching - could be enhanced with NLP
        query_terms = [term.lower().strip() for term in query.split() if len(term.strip()) > 2]
        content_lower = content.lower()

        matches = []
        for term in query_terms:
            if term in content_lower:
                matches.append(term)

        return matches

    def _generate_highlight(self, content: str, query: str, max_length: int = 200) -> str:
        """Generate highlighted snippet from content.

        Args:
            content: Full content text
            query: Query text to highlight
            max_length: Maximum length of highlight

        Returns:
            Highlighted text snippet
        """
        if not content:
            return ""

        # Simple highlighting - find best matching segment
        query_terms = [term.lower() for term in query.split() if len(term.strip()) > 2]
        content_lower = content.lower()

        # Find the segment with most query term matches
        best_segment = content[:max_length]
        best_score = 0

        # Check segments of content for query term density
        for i in range(0, len(content) - max_length + 1, 50):  # Slide window every 50 chars
            segment = content[i:i + max_length]
            segment_lower = segment.lower()

            # Count query term matches
            score = sum(1 for term in query_terms if term in segment_lower)

            if score > best_score:
                best_score = score
                best_segment = segment

        # Add ellipsis if truncated
        if len(best_segment) < len(content):
            if not best_segment.startswith(content[:10]):  # Not at start
                best_segment = "..." + best_segment
            if not best_segment.endswith(content[-10:]):  # Not at end
                best_segment = best_segment + "..."

        return best_segment

    def aggregate_multiple_searches(
        self,
        search_results: List[List[SearchResult]],
        method: str = "reciprocal_rank",
    ) -> List[SearchResult]:
        """Aggregate results from multiple searches.

        Args:
            search_results: List of search result lists
            method: Aggregation method ("reciprocal_rank", "average_score", "max_score")

        Returns:
            Aggregated search results
        """
        if not search_results:
            return []

        if method == "reciprocal_rank":
            return self._reciprocal_rank_fusion(search_results)
        elif method == "average_score":
            return self._average_score_aggregation(search_results)
        elif method == "max_score":
            return self._max_score_aggregation(search_results)
        else:
            # Default to reciprocal rank fusion
            return self._reciprocal_rank_fusion(search_results)

    def _reciprocal_rank_fusion(
        self,
        search_results: List[List[SearchResult]],
        k: int = 60,
    ) -> List[SearchResult]:
        """Perform reciprocal rank fusion on multiple result sets.

        Args:
            search_results: List of search result lists
            k: Fusion constant (typically 60)

        Returns:
            Fused search results
        """
        # Track scores by document ID
        score_map: Dict[str, tuple[float, SearchResult]] = {}

        for results in search_results:
            for result in results:
                doc_key = str(result.document_id)

                # Calculate reciprocal rank score
                rr_score = 1.0 / (k + result.rank)

                if doc_key in score_map:
                    current_score, current_result = score_map[doc_key]
                    new_score = current_score + rr_score
                    score_map[doc_key] = (new_score, current_result)
                else:
                    score_map[doc_key] = (rr_score, result)

        # Sort by aggregated score and create new results
        sorted_items = sorted(score_map.items(), key=lambda x: x[1][0], reverse=True)

        fused_results = []
        for idx, (doc_key, (score, original_result)) in enumerate(sorted_items):
            # Create new similarity score with aggregated score
            aggregated_score = SimilarityScore(
                score=min(score, 1.0),  # Cap at 1.0
                method=original_result.similarity_score.method,
                confidence=original_result.similarity_score.confidence,
                rank=idx + 1,
                metadata=original_result.similarity_score.metadata.copy(),
            )
            aggregated_score.metadata["aggregation_method"] = "reciprocal_rank"
            aggregated_score.metadata["aggregation_score"] = score

            # Create new result with aggregated score
            fused_result = SearchResult(
                result_id=original_result.result_id,
                chunk_id=original_result.chunk_id,
                document_id=original_result.document_id,
                content=original_result.content,
                similarity_score=aggregated_score,
                rank=idx + 1,
                document_metadata=original_result.document_metadata,
                chunk_metadata=original_result.chunk_metadata,
                highlight_text=original_result.highlight_text,
                context_before=original_result.context_before,
                context_after=original_result.context_after,
                created_at=original_result.created_at,
                metadata=original_result.metadata.copy(),
            )
            fused_results.append(fused_result)

        return fused_results

    async def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        include_metadata: bool = True,
    ) -> List[SearchResult]:
        """Execute complete semantic search workflow.

        Orchestrates query embedding, similarity search, and result processing
        to provide a complete semantic search solution.

        Args:
            query: Search query text
            top_k: Maximum number of results to return
            min_similarity: Minimum similarity threshold
            filters: Optional search filters
            use_cache: Whether to use query embedding cache
            include_metadata: Whether to include result metadata

        Returns:
            List of processed search results sorted by relevance

        Raises:
            RetrievalError: If search execution fails
        """
        try:
            # Step 1: Validate input
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")

            # Step 2: Embed the query
            embedding_result = await self._embedding_service.embed_query(
                query=query.strip(),
                use_cache=use_cache,
                store_embedding=use_cache,  # Store cached queries for future use
            )

            # Step 3: Create search query entity
            search_query = SearchQuery.create(
                text=query.strip(),
                top_k=top_k,
                min_similarity=min_similarity,
                include_metadata=include_metadata,
                filters=filters,
            )

            # Step 4: Validate search query
            await self._search_repository.validate_query(search_query)

            # Step 5: Perform similarity search
            raw_results = await self._search_repository.semantic_search(search_query)

            # Step 6: Process and rank results
            processed_results = await self.process_search_results(
                raw_results=raw_results,
                query=search_query,
                query_embedding_dimension=embedding_result.vector_dimension,
            )

            return processed_results

        except Exception as e:
            # Re-raise validation errors
            if "Query cannot be empty" in str(e) or "validation" in str(e).lower():
                raise
            # Wrap other errors in retrieval error
            raise Exception(f"Semantic search execution failed: {str(e)}") from e

    async def semantic_search_with_metrics(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        """Execute semantic search with detailed performance metrics.

        Args:
            query: Search query text
            top_k: Maximum number of results to return
            min_similarity: Minimum similarity threshold
            filters: Optional search filters
            use_cache: Whether to use query embedding cache
            include_metadata: Whether to include result metadata

        Returns:
            Dictionary with search results and performance metrics

        Raises:
            RetrievalError: If search execution fails
        """
        import time


        start_time = time.time()

        # Execute search
        results = await self.semantic_search(
            query=query,
            top_k=top_k,
            min_similarity=min_similarity,
            filters=filters,
            use_cache=use_cache,
            include_metadata=include_metadata,
        )

        # Calculate metrics
        total_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Calculate average similarity score
        avg_similarity = 0.0
        if results:
            avg_similarity = sum(result.similarity_score.score for result in results) / len(results)

        # Get embedding cache status
        embedding_cache_hit = False
        if use_cache:
            # We could track this from the embedding service, but for now use a simple heuristic
            embedding_cache_hit = total_time < 100  # If very fast, likely from cache

        return {
            "results": results,
            "metrics": {
                "total_processing_time_ms": total_time,
                "result_count": len(results),
                "avg_similarity_score": avg_similarity,
                "cache_hit": embedding_cache_hit,
                "query_length": len(query),
                "min_similarity_threshold": min_similarity,
                "top_k_requested": top_k,
                "top_k_returned": len(results),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        }

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
            List of similar chunks
        """
        return await self._search_repository.find_similar_chunks(
            chunk_ids=chunk_ids,
            top_k=top_k,
            min_similarity=min_similarity,
        )

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
        """
        try:
            if not partial_query or len(partial_query.strip()) < 2:
                return []

            return await self._search_repository.get_search_suggestions(
                partial_query=partial_query.strip(),
                max_suggestions=max_suggestions,
                min_similarity=min_similarity,
            )
        except Exception as e:
            # Return empty list on failure to avoid breaking UI
            return []

    async def get_retrieval_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for retrieval operations.

        Returns:
            Dictionary with retrieval service metrics
        """
        try:
            # Get embedding service health
            embedding_healthy = await self._embedding_service.health_check()

            # Get search repository health and stats
            search_health = await self._search_repository.health_check()
            search_stats = await self._search_repository.get_search_statistics()

            return {
                "embedding_service_healthy": embedding_healthy,
                "search_repository_healthy": search_health,
                "search_statistics": search_stats,
                "overall_status": "healthy" if embedding_healthy and search_health else "degraded",
            }
        except Exception as e:
            return {
                "error": f"Failed to get retrieval metrics: {str(e)}",
                "embedding_service_healthy": False,
                "search_repository_healthy": False,
                "search_statistics": {},
                "overall_status": "error",
            }

    def _average_score_aggregation(
        self,
        search_results: List[List[SearchResult]],
    ) -> List[SearchResult]:
        """Aggregate results using average similarity scores.

        Args:
            search_results: List of search result lists

        Returns:
            Aggregated search results
        """
        # Group results by document ID
        doc_groups: Dict[str, List[SearchResult]] = {}

        for results in search_results:
            for result in results:
                doc_key = str(result.document_id)
                if doc_key not in doc_groups:
                    doc_groups[doc_key] = []
                doc_groups[doc_key].append(result)

        # Calculate average scores and create aggregated results
        aggregated_results = []
        for doc_key, doc_results in doc_groups.items():
            avg_score = sum(r.similarity_score.score for r in doc_results) / len(doc_results)

            # Use the first result as base, update with average score
            base_result = doc_results[0]
            aggregated_score = SimilarityScore(
                score=avg_score,
                method=base_result.similarity_score.method,
                confidence=base_result.similarity_score.confidence,
                metadata=base_result.similarity_score.metadata.copy(),
            )
            aggregated_score.metadata["aggregation_method"] = "average_score"
            aggregated_score.metadata["result_count"] = len(doc_results)

            aggregated_result = SearchResult(
                result_id=base_result.result_id,
                chunk_id=base_result.chunk_id,
                document_id=base_result.document_id,
                content=base_result.content,
                similarity_score=aggregated_score,
                rank=0,  # Will be set after sorting
                document_metadata=base_result.document_metadata,
                chunk_metadata=base_result.chunk_metadata,
                highlight_text=base_result.highlight_text,
                context_before=base_result.context_before,
                context_after=base_result.context_after,
                created_at=base_result.created_at,
                metadata=base_result.metadata.copy(),
            )
            aggregated_results.append(aggregated_result)

        # Sort by average score and set ranks
        aggregated_results.sort(key=lambda r: r.similarity_score.score, reverse=True)
        for idx, result in enumerate(aggregated_results):
            result.similarity_score.rank = idx + 1
            result.rank = idx + 1

        return aggregated_results

    def _max_score_aggregation(
        self,
        search_results: List[List[SearchResult]],
    ) -> List[SearchResult]:
        """Aggregate results using maximum similarity scores.

        Args:
            search_results: List of search result lists

        Returns:
            Aggregated search results
        """
        # Keep best result per document ID
        best_results: Dict[str, SearchResult] = {}

        for results in search_results:
            for result in results:
                doc_key = str(result.document_id)

                if (doc_key not in best_results or
                    result.similarity_score.score > best_results[doc_key].similarity_score.score):
                    best_results[doc_key] = result

        # Sort by score and set ranks
        aggregated_results = list(best_results.values())
        aggregated_results.sort(key=lambda r: r.similarity_score.score, reverse=True)

        for idx, result in enumerate(aggregated_results):
            result.similarity_score.rank = idx + 1
            result.rank = idx + 1
            result.similarity_score.metadata["aggregation_method"] = "max_score"

        return aggregated_results