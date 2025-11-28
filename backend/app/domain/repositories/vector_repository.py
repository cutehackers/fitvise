"""Vector repository interface.

This module contains the VectorRepository abstract interface that defines
the contract for vector storage and retrieval operations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from app.domain.value_objects.vector_embedding import VectorEmbedding


class VectorSearchResult:
    """Result from vector similarity search."""

    def __init__(
        self,
        document: Any,
        similarity_score: float,
        metadata: Optional[Dict[str, Any]] = None,
        distance: Optional[float] = None,
    ) -> None:
        """Initialize vector search result.

        Args:
            document: The retrieved document
            similarity_score: Similarity score (0.0-1.0)
            metadata: Additional metadata
            distance: Optional distance metric
        """
        self.document = document
        self.similarity_score = similarity_score
        self.metadata = metadata or {}
        self.distance = distance


class VectorRepository(ABC):
    """Abstract repository interface for vector storage and retrieval operations.

    This interface defines the contract for vector database operations including
    similarity search, batch operations, and management functions.
    """

    @abstractmethod
    async def similarity_search(
        self,
        embedding: VectorEmbedding,
        threshold: float = 0.75,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        """Execute similarity search for documents.

        Args:
            embedding: Query embedding to search for
            threshold: Minimum similarity threshold (0.0-1.0)
            limit: Maximum number of results to return
            filters: Optional metadata filters
            namespace: Optional namespace for search

        Returns:
            List of search results with similarity scores

        Raises:
            RetrievalError: If search operation fails
        """
        pass

    @abstractmethod
    async def batch_similarity_search(
        self,
        embeddings: List[VectorEmbedding],
        threshold: float = 0.75,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[List[VectorSearchResult]]:
        """Execute similarity search for multiple embeddings.

        Args:
            embeddings: List of query embeddings to search for
            threshold: Minimum similarity threshold (0.0-1.0)
            limit: Maximum number of results per query
            filters: Optional metadata filters

        Returns:
            List of search result lists, one for each embedding

        Raises:
            RetrievalError: If batch search operation fails
        """
        pass

    @abstractmethod
    async def keyword_search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Execute keyword-based search.

        Args:
            query: Keyword search query
            limit: Maximum number of results to return
            filters: Optional metadata filters

        Returns:
            List of search results

        Raises:
            RetrievalError: If keyword search fails
        """
        pass

    @abstractmethod
    async def hybrid_search(
        self,
        vector_query: Optional[str] = None,
        keyword_query: Optional[str] = None,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Execute hybrid search combining vector and keyword search.

        Args:
            vector_query: Vector-based search query
            keyword_query: Keyword-based search query
            vector_weight: Weight for vector search results
            keyword_weight: Weight for keyword search results
            limit: Maximum number of results to return
            filters: Optional metadata filters

        Returns:
            List of combined search results

        Raises:
            RetrievalError: If hybrid search fails
        """
        pass

    @abstractmethod
    async def add_embeddings(
        self,
        embeddings: List[tuple[Any, VectorEmbedding, Dict[str, Any]]],
        namespace: Optional[str] = None,
    ) -> None:
        """Add multiple embeddings to the vector store.

        Args:
            embeddings: List of (document_id, embedding, metadata) tuples
            namespace: Optional namespace for the embeddings

        Raises:
            RetrievalError: If add operation fails
        """
        pass

    @abstractmethod
    async def update_embedding(
        self,
        document_id: str,
        embedding: VectorEmbedding,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> None:
        """Update an existing embedding.

        Args:
            document_id: ID of the document to update
            embedding: New embedding
            metadata: Optional new metadata
            namespace: Optional namespace

        Raises:
            RetrievalError: If update operation fails
        """
        pass

    @abstractmethod
    async def delete_embeddings(
        self,
        document_ids: List[str],
        namespace: Optional[str] = None,
    ) -> None:
        """Delete embeddings by document IDs.

        Args:
            document_ids: List of document IDs to delete
            namespace: Optional namespace

        Raises:
            RetrievalError: If delete operation fails
        """
        pass

    @abstractmethod
    async def get_embedding(
        self,
        document_id: str,
        namespace: Optional[str] = None,
    ) -> Optional[VectorEmbedding]:
        """Get embedding for a specific document.

        Args:
            document_id: ID of the document
            namespace: Optional namespace

        Returns:
            Vector embedding if found, None otherwise

        Raises:
            RetrievalError: If get operation fails
        """
        pass

    @abstractmethod
    async def get_document_metadata(
        self,
        document_id: str,
        namespace: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific document.

        Args:
            document_id: ID of the document
            namespace: Optional namespace

        Returns:
            Document metadata if found, None otherwise

        Raises:
            RetrievalError: If get operation fails
        """
        pass

    @abstractmethod
    async def list_namespaces(self) -> List[str]:
        """List all available namespaces.

        Returns:
            List of namespace names

        Raises:
            RetrievalError: If list operation fails
        """
        pass

    @abstractmethod
    async def create_namespace(
        self,
        namespace: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create a new namespace.

        Args:
            namespace: Name of the namespace to create
            config: Optional configuration for the namespace

        Raises:
            RetrievalError: If create operation fails
        """
        pass

    @abstractmethod
    async def delete_namespace(
        self,
        namespace: str,
    ) -> None:
        """Delete a namespace and all its embeddings.

        Args:
            namespace: Name of the namespace to delete

        Raises:
            RetrievalError: If delete operation fails
        """
        pass

    @abstractmethod
    async def get_namespace_stats(
        self,
        namespace: str,
    ) -> Dict[str, Any]:
        """Get statistics for a namespace.

        Args:
            namespace: Name of the namespace

        Returns:
            Dictionary with namespace statistics

        Raises:
            RetrievalError: If stats operation fails
        """
        pass

    @abstractmethod
    async def optimize_index(
        self,
        namespace: Optional[str] = None,
    ) -> None:
        """Optimize the vector index for better performance.

        Args:
            namespace: Optional namespace to optimize

        Raises:
            RetrievalError: If optimize operation fails
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the vector repository.

        Returns:
            Dictionary with health status information

        Raises:
            RetrievalError: If health check fails
        """
        pass

    @abstractmethod
    async def get_query_statistics(
        self,
        time_range_days: int = 7,
    ) -> Dict[str, Any]:
        """Get query execution statistics.

        Args:
            time_range_days: Number of days to look back for statistics

        Returns:
            Dictionary with query statistics

        Raises:
            RetrievalError: If stats operation fails
        """
        pass

    def supports_keyword_search(self) -> bool:
        """Check if this repository supports keyword search."""
        return False

    def supports_hybrid_search(self) -> bool:
        """Check if this repository supports hybrid search."""
        return False

    def supports_namespaces(self) -> bool:
        """Check if this repository supports namespaces."""
        return False

    def supports_batch_operations(self) -> bool:
        """Check if this repository supports batch operations."""
        return True

    def supports_metadata_filters(self) -> bool:
        """Check if this repository supports metadata filtering."""
        return True

    def get_supported_distance_metrics(self) -> List[str]:
        """Get list of supported distance metrics."""
        return ["cosine"]

    def get_max_batch_size(self) -> int:
        """Get maximum batch size for operations."""
        return 1000

    def get_timeout_seconds(self) -> int:
        """Get default timeout for operations."""
        return 30