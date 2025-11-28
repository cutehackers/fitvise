"""Retrieval result domain entity.

This module contains the RetrievalResult entity that encapsulates
results from vector-based retrieval operations with business logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Tuple

from app.domain.entities.document_reference import DocumentReference
from app.domain.entities.vector_retrieval_config import VectorRetrievalConfig
from app.domain.value_objects.similarity_score import SimilarityScore


@dataclass
class RetrievalResult:
    """Result entity from vector-based retrieval operations.

    This entity contains the results of a retrieval operation along with
    business methods for processing and analyzing the results.
    """

    documents: List[DocumentReference]
    similarity_scores: List[float]
    query_id: str
    config: VectorRetrievalConfig
    total_searched: int = 0
    search_time_ms: float = 0.0
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    reranking_applied: bool = False
    cache_hit: bool = False

    def __post_init__(self):
        """Validate result consistency after initialization."""
        if len(self.documents) != len(self.similarity_scores):
            raise ValueError(
                f"Documents count ({len(self.documents)}) must match "
                f"scores count ({len(self.similarity_scores)})"
            )

        if not self.query_id:
            raise ValueError("Query ID cannot be empty")

        if self.total_searched < len(self.documents):
            raise ValueError(
                f"Total searched ({self.total_searched}) cannot be less than "
                f"returned documents ({len(self.documents)})"
            )

    @property
    def result_count(self) -> int:
        """Get the number of documents in this result."""
        return len(self.documents)

    @property
    def is_empty(self) -> bool:
        """Check if the result contains any documents."""
        return self.result_count == 0

    @property
    def average_similarity(self) -> float:
        """Calculate average similarity score."""
        if self.is_empty:
            return 0.0
        return sum(self.similarity_scores) / len(self.similarity_scores)

    @property
    def max_similarity(self) -> float:
        """Get the maximum similarity score."""
        return max(self.similarity_scores) if self.similarity_scores else 0.0

    @property
    def min_similarity(self) -> float:
        """Get the minimum similarity score."""
        return min(self.similarity_scores) if self.similarity_scores else 0.0

    @property
    def precision_at_k(self, k: int = 5) -> float:
        """Calculate precision at k (assumes all results are relevant)."""
        if k <= 0:
            return 0.0

        relevant_at_k = min(k, self.result_count)
        return relevant_at_k / min(k, self.total_searched)

    @property
    def recall_at_k(self, k: int = 10) -> float:
        """Calculate recall at k."""
        if k <= 0 or self.total_searched == 0:
            return 0.0

        retrieved_at_k = min(k, self.result_count)
        return retrieved_at_k / self.total_searched

    def get_top_results(self, n: int) -> List[Tuple[DocumentReference, float]]:
        """Get the top N results with their scores."""
        if n <= 0:
            return []

        # Combine documents with scores and sort by score (descending)
        scored_docs = list(zip(self.documents, self.similarity_scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return scored_docs[:n]

    def get_documents_above_threshold(
        self, threshold: float = None
    ) -> List[Tuple[DocumentReference, float]]:
        """Get documents with similarity above the specified threshold."""
        if threshold is None:
            threshold = self.config.similarity_threshold

        scored_docs = list(zip(self.documents, self.similarity_scores))
        return [
            (doc, score) for doc, score in scored_docs
            if score >= threshold
        ]

    def get_high_relevance_documents(
        self, threshold: float = 0.8
    ) -> List[Tuple[DocumentReference, float]]:
        """Get documents with high relevance scores."""
        return self.get_documents_above_threshold(threshold)

    def get_diverse_results(
        self, max_results: int = None, similarity_threshold: float = 0.9
    ) -> List[Tuple[DocumentReference, float]]:
        """Get diverse results by removing near-duplicates."""
        if max_results is None:
            max_results = self.config.max_results

        diverse_results = []
        scored_docs = list(zip(self.documents, self.similarity_scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        for doc, score in scored_docs:
            if len(diverse_results) >= max_results:
                break

            # Check for content similarity with existing results
            is_duplicate = False
            for existing_doc, _ in diverse_results:
                content_similarity = self._calculate_content_similarity(
                    doc.content, existing_doc.content
                )
                if content_similarity > similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                diverse_results.append((doc, score))

        return diverse_results

    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate simple content similarity (placeholder for more sophisticated algorithm)."""
        # Simple word overlap similarity
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def filter_by_metadata(
        self, filter_criteria: Dict[str, Any]
    ) -> RetrievalResult:
        """Create a new result filtered by metadata criteria."""
        filtered_docs = []
        filtered_scores = []

        for doc, score in zip(self.documents, self.similarity_scores):
            if self._matches_metadata_filter(doc.metadata, filter_criteria):
                filtered_docs.append(doc)
                filtered_scores.append(score)

        return RetrievalResult(
            documents=filtered_docs,
            similarity_scores=filtered_scores,
            query_id=self.query_id,
            config=self.config,
            total_searched=self.total_searched,
            search_time_ms=self.search_time_ms,
            processing_time_ms=self.processing_time_ms,
            metadata={
                **self.metadata,
                "filtered_by": filter_criteria,
                "original_result_count": self.result_count,
            },
            timestamp=self.timestamp,
            reranking_applied=self.reranking_applied,
            cache_hit=self.cache_hit,
        )

    def _matches_metadata_filter(
        self, metadata: Dict[str, Any], criteria: Dict[str, Any]
    ) -> bool:
        """Check if metadata matches the filter criteria."""
        for key, expected_value in criteria.items():
            if key not in metadata:
                return False

            actual_value = metadata[key]
            if isinstance(expected_value, (list, set)):
                if actual_value not in expected_value:
                    return False
            elif actual_value != expected_value:
                return False

        return True

    def apply_business_rules(self) -> RetrievalResult:
        """Apply business rules to improve result quality."""
        # Apply minimum threshold filtering
        threshold_docs = self.get_documents_above_threshold(self.config.similarity_threshold)
        docs, scores = zip(*threshold_docs) if threshold_docs else ([], [])

        # Apply boosting for recent documents if configured
        if self.config.should_boost_recent_documents():
            boosted_scores = []
            for doc, score in zip(docs, scores):
                is_recent = self._is_recent_document(doc)
                boosted_score = self.config.apply_boost_factor(score, is_recent)
                boosted_scores.append(boosted_score)
            scores = boosted_scores

        # Re-sort by boosted scores
        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        if scored_docs:
            docs, scores = zip(*scored_docs)

        return RetrievalResult(
            documents=list(docs),
            similarity_scores=list(scores),
            query_id=self.query_id,
            config=self.config,
            total_searched=self.total_searched,
            search_time_ms=self.search_time_ms,
            processing_time_ms=self.processing_time_ms,
            metadata={
                **self.metadata,
                "business_rules_applied": True,
                "threshold_applied": self.config.similarity_threshold,
                "boosting_applied": self.config.should_boost_recent_documents(),
            },
            timestamp=self.timestamp,
            reranking_applied=self.reranking_applied,
            cache_hit=self.cache_hit,
        )

    def _is_recent_document(self, doc: DocumentReference) -> bool:
        """Check if a document is considered recent."""
        # Simple heuristic: check if document has a timestamp in metadata
        if "created_at" in doc.metadata:
            created_at = doc.metadata["created_at"]
            if isinstance(created_at, datetime):
                time_diff = datetime.utcnow() - created_at
                return time_diff.days <= 30  # Documents from last 30 days

        # Check if document has a "recent" flag
        return doc.metadata.get("is_recent", False)

    def calculate_quality_score(self) -> float:
        """Calculate overall quality score for this result."""
        if self.is_empty:
            return 0.0

        # Factors contributing to quality:
        # 1. Average similarity score (40% weight)
        similarity_factor = self.average_similarity * 0.4

        # 2. Result count appropriateness (20% weight)
        # Penalize too few or too many results
        ideal_count = self.config.max_results
        count_factor = 1.0 - abs(self.result_count - ideal_count) / ideal_count * 0.5
        count_factor = max(0.0, min(1.0, count_factor)) * 0.2

        # 3. Score distribution (20% weight)
        # Prefer results with good distribution of scores
        if self.result_count > 1:
            score_variance = sum((s - self.average_similarity) ** 2 for s in self.similarity_scores) / self.result_count
            distribution_factor = (1.0 - min(score_variance, 1.0)) * 0.2
        else:
            distribution_factor = 0.1  # Small penalty for single result

        # 4. Processing efficiency (10% weight)
        # Prefer faster results
        time_factor = max(0.0, 1.0 - self.processing_time_ms / 1000.0) * 0.1

        # 5. Cache efficiency bonus (10% weight)
        cache_factor = 0.1 if self.cache_hit else 0.0

        total_quality = similarity_factor + count_factor + distribution_factor + time_factor + cache_factor
        return min(1.0, max(0.0, total_quality))

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "documents": [doc.to_dict() for doc in self.documents],
            "similarity_scores": self.similarity_scores,
            "query_id": self.query_id,
            "config": self.config.to_dict(),
            "total_searched": self.total_searched,
            "search_time_ms": self.search_time_ms,
            "processing_time_ms": self.processing_time_ms,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "reranking_applied": self.reranking_applied,
            "cache_hit": self.cache_hit,
            "result_count": self.result_count,
            "average_similarity": self.average_similarity,
            "max_similarity": self.max_similarity,
            "min_similarity": self.min_similarity,
            "quality_score": self.calculate_quality_score(),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get a concise summary of the result."""
        return {
            "query_id": self.query_id,
            "result_count": self.result_count,
            "total_searched": self.total_searched,
            "average_similarity": round(self.average_similarity, 3),
            "max_similarity": round(self.max_similarity, 3),
            "quality_score": round(self.calculate_quality_score(), 3),
            "processing_time_ms": round(self.processing_time_ms, 2),
            "reranking_applied": self.reranking_applied,
            "cache_hit": self.cache_hit,
            "high_relevance_count": len(self.get_high_relevance_documents()),
        }