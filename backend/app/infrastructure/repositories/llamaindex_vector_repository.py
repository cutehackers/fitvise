"""LlamaIndex vector repository implementation.

This module contains the LlamaIndexVectorRepository implementation that bridges
our DDD domain interface with LlamaIndex's vector operations.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from app.domain.repositories.vector_repository import (
    VectorRepository,
    VectorSearchResult,
)
from app.domain.exceptions.retrieval_exceptions import (
    RetrievalError,
    SearchExecutionError,
    SimilaritySearchError,
)
from app.domain.value_objects.vector_embedding import VectorEmbedding
from app.infrastructure.external_services.vector_stores.weaviate_client import (
    WeaviateClient,
)

# Import LlamaIndex components
try:
    from llama_index.core import VectorStoreIndex
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.weaviate import WeaviateVectorStore
    from llama_index.core.schema import Document
except ImportError as e:
    raise RetrievalError(
        f"LlamaIndex components not available: {str(e)}",
        operation="initialization"
    ) from e

logger = logging.getLogger(__name__)


class LlamaIndexVectorRepository(VectorRepository):
    """LlamaIndex-based implementation of VectorRepository.

    This repository wraps LlamaIndex's vector operations to provide
    our domain interface while maintaining the performance benefits of
    direct LlamaIndex access.
    """

    def __init__(
        self,
        index_name: str = "Chunk",
        text_key: str = "text",
        embed_model_name: str = "Alibaba-NLP/gte-multilingual-base",
        weaviate_client: Optional[WeaviateClient] = None,
        index: Optional[VectorStoreIndex] = None,
        embed_model: Optional[Any] = None,
    ) -> None:
        """Initialize LlamaIndex vector repository.

        Args:
            index_name: Weaviate collection name
            text_key: Property name for document text
            embed_model_name: HuggingFace embedding model name
            weaviate_client: Connected Weaviate client (optional if index provided)
            index: Pre-existing LlamaIndex index (optional)
            embed_model: Pre-existing embedding model (optional)
        """
        self._index_name = index_name
        self._text_key = text_key
        self._embed_model_name = embed_model_name
        self._weaviate_client = weaviate_client

        # Initialize LlamaIndex components
        if index is not None:
            self._index = index
            self._embed_model = embed_model or HuggingFaceEmbedding(
                model_name=embed_model_name,
                trust_remote_code=True,
            )
        else:
            if weaviate_client is None:
                raise RetrievalError(
                    "Either weaviate_client or index must be provided",
                    operation="initialization"
                )

            if not weaviate_client.is_connected:
                raise RetrievalError(
                    "WeaviateClient must be connected before use",
                    operation="initialization"
                )

            self._initialize_llama_index_components()

        self._retriever: Optional[VectorIndexRetriever] = None

    def _initialize_llama_index_components(self) -> None:
        """Initialize LlamaIndex vector store and index."""
        try:
            # Create Weaviate vector store with LlamaIndex
            vector_store = WeaviateVectorStore(
                weaviate_client=self._weaviate_client._client,
                index_name=self._index_name,
                text_key=self._text_key,
            )

            # Create embedding model
            self._embed_model = HuggingFaceEmbedding(
                model_name=self._embed_model_name,
                trust_remote_code=True,
            )

            # Create index from existing vector store (no re-indexing)
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=self._embed_model,
            )

            logger.info(
                "LlamaIndex components initialized: index=%s, model=%s",
                self._index_name,
                self._embed_model_name
            )

        except Exception as e:
            raise RetrievalError(
                f"Failed to initialize LlamaIndex components: {str(e)}",
                operation="initialization"
            ) from e

    def _get_retriever(self, similarity_top_k: int = 10) -> VectorIndexRetriever:
        """Get or create LlamaIndex retriever."""
        if self._retriever is None or self._retriever._similarity_top_k != similarity_top_k:
            self._retriever = self._index.as_retriever(
                similarity_top_k=similarity_top_k
            )
        return self._retriever

    async def similarity_search(
        self,
        embedding: VectorEmbedding,
        threshold: float = 0.75,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        """Execute similarity search for documents."""
        try:
            start_time = datetime.utcnow()

            # Create LlamaIndex retriever
            retriever = self._get_retriever(similarity_top_k=limit)

            # Convert embedding to numpy array for LlamaIndex
            import numpy as np
            embedding_array = np.array(embedding.values, dtype=np.float32)

            # Apply threshold if specified
            doc_ids = None
            if threshold > 0:
                # Convert threshold to distance (LlamaIndex uses distance)
                # For cosine similarity, distance = 1 - similarity
                max_distance = 1.0 - threshold
                doc_ids = await self._get_doc_ids_by_distance_threshold(max_distance)

            # Apply metadata filters if provided
            if filters:
                doc_ids = await self._apply_metadata_filters(filters, doc_ids)

            # Configure retriever with filters
            if doc_ids:
                retriever = self._index.as_retriever(
                    similarity_top_k=limit,
                    doc_ids=doc_ids
                )

            # Execute retrieval using embedding
            nodes = retriever.retrieve(embedding_array)

            # Convert results to domain format
            results = []
            for node in nodes:
                # Convert LlamaIndex score (distance) to similarity score
                distance = node.score if node.score is not None else 1.0
                similarity = 1.0 - max(0.0, min(1.0, distance))

                # Apply threshold filtering
                if similarity < threshold:
                    continue

                # Create domain search result
                document = self._convert_node_to_document(node)
                result = VectorSearchResult(
                    document=document,
                    similarity_score=similarity,
                    metadata=self._extract_node_metadata(node),
                    distance=distance,
                )
                results.append(result)

            search_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.debug(
                "LlamaIndex similarity search completed: %d results in %.2fms",
                len(results),
                search_time
            )

            return results[:limit]  # Ensure we don't exceed the requested limit

        except Exception as e:
            logger.error("LlamaIndex similarity search failed: %s", str(e))
            raise SimilaritySearchError(
                f"Similarity search failed: {str(e)}",
                vector_dimension=embedding.dimension,
                similarity_method="cosine"
            ) from e

    async def batch_similarity_search(
        self,
        embeddings: List[VectorEmbedding],
        threshold: float = 0.75,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[List[VectorSearchResult]]:
        """Execute similarity search for multiple embeddings."""
        if not self.supports_batch_operations():
            raise RetrievalError(
                "Batch operations not supported for this repository",
                operation="batch_similarity_search"
            )

        results = []
        for embedding in embeddings:
            try:
                search_results = await self.similarity_search(
                    embedding=embedding,
                    threshold=threshold,
                    limit=limit,
                    filters=filters,
                )
                results.append(search_results)
            except Exception as e:
                logger.error(f"Failed to search embedding {embedding.dimension}D: {str(e)}")
                results.append([])  # Return empty list for failed search

        return results

    async def keyword_search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Execute keyword-based search."""
        # LlamaIndex doesn't natively support keyword search
        # This could be implemented using BM25 or other text search methods
        # For now, we'll return empty results
        logger.warning("Keyword search not implemented for LlamaIndex repository")
        return []

    async def hybrid_search(
        self,
        vector_query: Optional[str] = None,
        keyword_query: Optional[str] = None,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Execute hybrid search combining vector and keyword search."""
        if not self.supports_hybrid_search():
            raise RetrievalError(
                "Hybrid search not supported for this repository",
                operation="hybrid_search"
            )

        # For now, fall back to vector-only search
        if vector_query:
            # Generate embedding for vector query
            embed_model = HuggingFaceEmbedding(
                model_name=self._embed_model_name,
                trust_remote_code=True,
            )
            import numpy as np

            # Generate embedding (this is sync in HuggingFace)
            query_embedding = embed_model.get_text_embedding(vector_query)
            embedding_obj = VectorEmbedding.from_numpy(
                np.array(query_embedding),
                model_name=self._embed_model_name
            )

            return await self.similarity_search(
                embedding=embedding_obj,
                limit=limit,
                filters=filters,
            )
        else:
            return []

    async def add_embeddings(
        self,
        embeddings: List[tuple[Any, VectorEmbedding, Dict[str, Any]]],
        namespace: Optional[str] = None,
    ) -> None:
        """Add multiple embeddings to the vector store."""
        try:
            # Convert to LlamaIndex Document objects
            documents = []
            for doc_id, embedding, metadata in embeddings:
                # Create LlamaIndex Document
                text = metadata.get("text", "") or ""
                llama_doc = Document(
                    text=text,
                    doc_id=str(doc_id),
                    metadata={
                        **metadata,
                        "embedding_dimension": embedding.dimension,
                        "model_name": embedding.model_name,
                    }
                )
                documents.append(llama_doc)

            # Add documents to index
            if documents:
                self._index.insert(documents)

            logger.info(f"Added {len(documents)} embeddings to LlamaIndex")

        except Exception as e:
            logger.error(f"Failed to add embeddings to LlamaIndex: {str(e)}")
            raise RetrievalError(
                f"Failed to add embeddings: {str(e)}",
                operation="add_embeddings"
            ) from e

    async def update_embedding(
        self,
        document_id: str,
        embedding: VectorEmbedding,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> None:
        """Update an existing embedding."""
        # LlamaIndex doesn't have direct update functionality
        # For now, we would need to implement this as delete + add
        await self.delete_embeddings([document_id], namespace)
        await self.add_embeddings([(document_id, embedding, metadata or {})])

    async def delete_embeddings(
        self,
        document_ids: List[str],
        namespace: Optional[str] = None,
    ) -> None:
        """Delete embeddings by document IDs."""
        try:
            # LlamaIndex delete by doc_id
            for doc_id in document_ids:
                try:
                    self._index.delete(doc_id=doc_id)
                except Exception as e:
                    logger.warning(f"Failed to delete document {doc_id}: {str(e)}")
                    # Continue with other deletions

            logger.info(f"Deleted {len(document_ids)} embeddings from LlamaIndex")

        except Exception as e:
            logger.error(f"Failed to delete embeddings from LlamaIndex: {str(e)}")
            raise RetrievalError(
                f"Failed to delete embeddings: {str(e)}",
                operation="delete_embeddings"
            ) from e

    async def get_embedding(
        self,
        document_id: str,
        namespace: Optional[str] = None,
    ) -> Optional[VectorEmbedding]:
        """Get embedding for a specific document."""
        try:
            # Get document by ID
            docstore = self._index.docstore
            node = docstore.get_document(document_id)

            if node is None:
                return None

            # Get text from node
            text = node.get_content()

            # Generate embedding for the text
            embed_model = HuggingFaceEmbedding(
                model_name=self._embed_model_name,
                trust_remote_code=True,
            )

            # Generate embedding (sync)
            embedding_array = embed_model.get_text_embedding(text)
            return VectorEmbedding.from_numpy(
                embedding_array,
                model_name=self._embed_model_name,
                metadata={
                    "document_id": document_id,
                    "node_metadata": node.metadata,
                }
            )

        except Exception as e:
            logger.error(f"Failed to get embedding for {document_id}: {str(e)}")
            raise RetrievalError(
                f"Failed to get embedding: {str(e)}",
                operation="get_embedding"
            ) from e

    async def get_document_metadata(
        self,
        document_id: str,
        namespace: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific document."""
        try:
            docstore = self._index.docstore
            node = docstore.get_document(document_id)

            if node is None:
                return None

            return dict(node.metadata) if node.metadata else {}

        except Exception as e:
            logger.error(f"Failed to get metadata for {document_id}: {str(e)}")
            raise RetrievalError(
                f"Failed to get metadata: {str(e)}",
                operation="get_document_metadata"
            ) from e

    async def list_namespaces(self) -> List[str]:
        """List all available namespaces."""
        # LlamaIndex doesn't have explicit namespace support
        # We could implement this using collections if needed
        return []

    async def create_namespace(
        self,
        namespace: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create a new namespace."""
        raise NotImplementedError(
            "Namespace creation not supported by LlamaIndex repository"
        )

    async def delete_namespace(
        self,
        namespace: str,
    ) -> None:
        """Delete a namespace and all its embeddings."""
        raise NotImplementedError(
            "Namespace deletion not supported by LlamaIndex repository"
        )

    async def get_namespace_stats(
        self,
        namespace: str,
    ) -> Dict[str, Any]:
        """Get statistics for a namespace."""
        raise NotImplementedError(
            "Namespace stats not supported by LlamaIndex repository"
        )

    async def optimize_index(
        self,
        namespace: Optional[str] = None,
    ) -> None:
        """Optimize the vector index for better performance."""
        try:
            # LlamaIndex can optimize the index
            self._index.vector_store.optimize()
            logger.info("LlamaIndex index optimization completed")

        except Exception as e:
            logger.error(f"Failed to optimize LlamaIndex index: {str(e)}")
            raise RetrievalError(
                f"Failed to optimize index: {str(e)}",
                operation="optimize_index"
            ) from e

    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the vector repository."""
        try:
            health = {
                "status": "healthy",
                "repository_type": "llamaindex",
                "index_name": self._index_name,
                "model_name": self._embed_model_name,
                "weaviate_connected": self._weaviate_client.is_connected if self._weaviate_client else True,
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Test embedding generation
            try:
                test_text = "health check"
                embed_model = HuggingFaceEmbedding(
                    model_name=self._embed_model_name,
                    trust_remote_code=True,
                )
                test_embedding = embed_model.get_text_embedding(test_text)
                health["embedding_generation"] = "operational"
                health["embedding_dimension"] = len(test_embedding)
            except Exception as e:
                health["embedding_generation"] = "error"
                health["embedding_error"] = str(e)

            # Test basic retrieval
            try:
                import numpy as np
                test_embedding = np.random.rand(384).astype(np.float32)  # Common embedding size
                retriever = self._get_retriever(similarity_top_k=1)
                nodes = retriever.retrieve(test_embedding)
                health["basic_retrieval"] = "operational"
            except Exception as e:
                health["basic_retrieval"] = "error"
                health["retrieval_error"] = str(e)

            return health

        except Exception as e:
            return {
                "status": "error",
                "repository_type": "llamaindex",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    async def get_query_statistics(
        self,
        time_range_days: int = 7,
    ) -> Dict[str, Any]:
        """Get query execution statistics."""
        # LlamaIndex doesn't have built-in query statistics
        # This would need to be implemented with custom tracking
        return {
            "total_queries": 0,
            "average_response_time_ms": 0.0,
            "cache_hit_rate": 0.0,
            "average_similarity_score": 0.0,
            "top_search_terms": [],
            "error_rate": 0.0,
            "repository_type": "llamaindex",
            "statistics_available": False,
        }

    def supports_keyword_search(self) -> bool:
        """Check if this repository supports keyword search."""
        return False

    def supports_hybrid_search(self) -> bool:
        """Check if this repository supports hybrid search."""
        return False  # Limited to vector-only search

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
        return 100

    def get_timeout_seconds(self) -> int:
        """Get default timeout for operations."""
        return 30

    def _convert_node_to_document(self, node: Any) -> Any:
        """Convert LlamaIndex node to domain document format."""
        # This would convert LlamaIndex node to your DocumentReference format
        # Implementation depends on your specific domain entities

        # Placeholder implementation
        return {
            "id": node.node_id,
            "content": node.get_content(),
            "metadata": dict(node.metadata) if node.metadata else {},
            "source_uri": node.metadata.get("source_uri") if node.metadata else None,
            "chunk_index": node.metadata.get("chunk_index", 0),
        }

    def _extract_node_metadata(self, node: Any) -> Dict[str, Any]:
        """Extract metadata from LlamaIndex node."""
        metadata = {}
        if node.metadata:
            metadata.update(node.metadata)
        if hasattr(node, 'score') and node.score is not None:
            metadata["score"] = node.score
        return metadata

    async def _get_doc_ids_by_distance_threshold(
        self,
        max_distance: float,
    ) -> Optional[List[str]]:
        """Get document IDs that are within a distance threshold."""
        # This is a complex operation that would need efficient indexing
        # For now, return None to disable distance filtering
        return None

    async def _apply_metadata_filters(
        self,
        filters: Dict[str, Any],
        doc_ids: Optional[List[str]] = None,
    ) -> Optional[List[str]]:
        """Apply metadata filters to get document IDs."""
        # This would need to be implemented based on your specific filter requirements
        # For now, return the existing doc_ids unchanged
        return doc_ids