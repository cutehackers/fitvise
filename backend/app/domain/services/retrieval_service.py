"""Retrieval service for coordinating search operations (Task 2.4.1).

This module defines the RetrievalService domain service that coordinates
search result processing, ranking, and formatting for semantic search.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.domain.entities.search_query import SearchQuery
from app.domain.entities.search_result import SearchResult
from app.domain.value_objects.similarity_score import SimilarityScore


class RetrievalService:
    """Domain service for coordinating retrieval operations.

    Provides business logic for processing search results, applying ranking
    algorithms, and formatting results for presentation.
    """

    def __init__(self) -> None:
        """Initialize retrieval service."""
        pass

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