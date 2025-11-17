"""Weaviate embedding repository implementation (Task 2.3.3).

This module implements the EmbeddingRepository interface using Weaviate
vector database with support for batch operations and similarity search.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Sequence
from uuid import UUID

from app.domain.entities.embedding import Embedding
from app.domain.exceptions.embedding_exceptions import EmbeddingStorageError
from app.domain.repositories.embedding_repository import EmbeddingRepository
from app.domain.value_objects.embedding_vector import EmbeddingVector
from app.infrastructure.external_services.vector_stores.weaviate_client import (
    WeaviateClient,
)

logger = logging.getLogger(__name__)


class WeaviateEmbeddingRepository(EmbeddingRepository):
    """Weaviate-based embedding repository (Task 2.3.3).

    Implements EmbeddingRepository interface using Weaviate vector database
    with support for batch operations (10K chunks/hour target) and
    efficient similarity search.

    Examples:
        >>> client = WeaviateClient(config)
        >>> await client.connect()
        >>> repository = WeaviateEmbeddingRepository(client)
        >>> await repository.save(embedding)

        >>> # Batch save
        >>> await repository.batch_save(embeddings, batch_size=100)

        >>> # Similarity search
        >>> results = await repository.similarity_search(
        ...     query_vector=query.vector,
        ...     k=10,
        ...     filters={"doc_type": "pdf"}
        ... )
    """

    CLASS_NAME = "Chunk"

    def __init__(self, client: WeaviateClient) -> None:
        """Initialize Weaviate embedding repository.

        Args:
            client: Weaviate client instance
        """
        self._client = client

    async def save(self, embedding: Embedding) -> None:
        """Save a single embedding.

        Args:
            embedding: Embedding to save

        Raises:
            EmbeddingStorageError: If save operation fails
        """
        properties = self._embedding_to_properties(embedding)
        vector = embedding.vector.to_list()

        try:
            await self._client.create_object(
                class_name=self.CLASS_NAME,
                properties=properties,
                vector=vector,
                uuid=str(embedding.id),
            )

            logger.debug(f"Saved embedding {embedding.id}")

        except Exception as e:
            raise EmbeddingStorageError(
                message="Failed to save embedding",
                operation="save",
                embedding_id=embedding.id,
                details=str(e),
            ) from e

    async def batch_save(
        self,
        embeddings: Sequence[Embedding],
        batch_size: int = 100,
    ) -> int:
        """Save multiple embeddings in batches.

        Args:
            embeddings: Sequence of embeddings to save
            batch_size: Number of embeddings per batch

        Returns:
            Number of embeddings successfully saved

        Raises:
            EmbeddingStorageError: If batch save operation fails
        """
        if not embeddings:
            return 0

        try:
            logger.info(f"Batch saving {len(embeddings)} embeddings")

            # Prepare batch data
            properties_list = [
                self._embedding_to_properties(emb) for emb in embeddings
            ]
            vectors_list = [emb.vector.to_list() for emb in embeddings]
            uuids_list = [str(emb.id) for emb in embeddings]

            # Batch create
            count = await self._client.batch_create_objects(
                class_name=self.CLASS_NAME,
                objects=properties_list,
                vectors=vectors_list,
                uuids=uuids_list,
                batch_size=batch_size,
            )

            logger.info(f"Successfully saved {count} embeddings")
            return count

        except Exception as e:
            raise EmbeddingStorageError(
                message=f"Failed to batch save {len(embeddings)} embeddings",
                operation="batch_save",
                details=str(e),
            ) from e

    async def find_by_id(self, embedding_id: UUID) -> Optional[Embedding]:
        """Find embedding by ID.

        Args:
            embedding_id: Embedding identifier

        Returns:
            Embedding if found, None otherwise

        Raises:
            EmbeddingStorageError: If retrieval operation fails
        """
        try:
            obj = await self._client.get_object(
                class_name=self.CLASS_NAME,
                uuid=str(embedding_id),
                include_vector=True,
            )

            if obj is None:
                return None

            return self._object_to_embedding(obj, embedding_id)

        except Exception as e:
            raise EmbeddingStorageError(
                message=f"Failed to find embedding {embedding_id}",
                operation="find_by_id",
                embedding_id=embedding_id,
                details=str(e),
            ) from e

    async def find_by_chunk_id(self, chunk_id: UUID) -> Optional[Embedding]:
        """Find embedding by chunk ID.

        Args:
            chunk_id: Chunk identifier

        Returns:
            Embedding if found, None otherwise

        Raises:
            EmbeddingStorageError: If retrieval operation fails
        """
        embeddings = await self.find_by_chunk_ids([chunk_id])
        return embeddings[0] if embeddings else None

    async def find_by_chunk_ids(
        self, chunk_ids: Sequence[UUID]
    ) -> List[Embedding]:
        """Find embeddings by multiple chunk IDs.

        Args:
            chunk_ids: Sequence of chunk identifiers

        Returns:
            List of embeddings found

        Raises:
            EmbeddingStorageError: If retrieval operation fails
        """
        if not chunk_ids:
            return []

        try:
            # Build filter for chunk_ids
            filters = {
                "operator": "Or",
                "operands": [
                    {
                        "path": ["chunk_id"],
                        "operator": "Equal",
                        "valueString": str(cid),
                    }
                    for cid in chunk_ids
                ],
            }

            # Query with filter
            results = await self._client.similarity_search(
                class_name=self.CLASS_NAME,
                vector=[0.0] * 768,  # Dummy vector for filter-only query
                limit=len(chunk_ids),
                filters=filters,
                min_certainty=0.0,
                include_vector=True,
            )

            return [
                self._object_to_embedding(
                    obj, UUID(obj["properties"]["chunk_id"])
                )
                for obj in results
            ]

        except Exception as e:
            raise EmbeddingStorageError(
                message=f"Failed to find embeddings for {len(chunk_ids)} chunks",
                operation="find_by_chunk_ids",
                details=str(e),
            ) from e

    async def find_by_document_id(self, document_id: UUID) -> List[Embedding]:
        """Find all embeddings for a document.

        Args:
            document_id: Document identifier

        Returns:
            List of embeddings for the document

        Raises:
            EmbeddingStorageError: If retrieval operation fails
        """
        try:
            # Build filter for document_id
            filters = {
                "path": ["document_id"],
                "operator": "Equal",
                "valueString": str(document_id),
            }

            # Query with filter
            results = await self._client.similarity_search(
                class_name=self.CLASS_NAME,
                vector=[0.0] * 768,  # Dummy vector for filter-only query
                limit=10000,  # Large limit for all chunks
                filters=filters,
                min_certainty=0.0,
                include_vector=True,
            )

            return [
                self._object_to_embedding(
                    obj, UUID(obj["properties"]["chunk_id"])
                )
                for obj in results
            ]

        except Exception as e:
            raise EmbeddingStorageError(
                message=f"Failed to find embeddings for document {document_id}",
                operation="find_by_document_id",
                details=str(e),
            ) from e

    async def similarity_search(
        self,
        query_vector: EmbeddingVector,
        k: int = 10,
        filters: Optional[Dict[str, any]] = None,
        min_similarity: float = 0.0,
    ) -> List[tuple[Embedding, float]]:
        """Search for similar embeddings using vector similarity.

        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            filters: Optional metadata filters
            min_similarity: Minimum similarity threshold

        Returns:
            List of (embedding, similarity_score) tuples, sorted by similarity

        Raises:
            EmbeddingStorageError: If search operation fails
        """
        try:
            # Convert similarity to certainty (Weaviate uses certainty 0-1)
            min_certainty = min_similarity

            # Execute search
            results = await self._client.similarity_search(
                class_name=self.CLASS_NAME,
                vector=query_vector.to_list(),
                limit=k,
                filters=self._build_weaviate_filters(filters) if filters else None,
                min_certainty=min_certainty,
                include_vector=True,
            )

            # Convert to embeddings with scores
            search_results = []
            for obj in results:
                additional = obj.get("_additional", {})
                certainty = additional.get("certainty", 0.0)

                embedding = self._object_to_embedding(
                    obj, UUID(obj["properties"]["chunk_id"])
                )
                search_results.append((embedding, certainty))

            return search_results

        except Exception as e:
            raise EmbeddingStorageError(
                message="Failed to perform similarity search",
                operation="similarity_search",
                details=str(e),
            ) from e

    async def exists(self, embedding_id: UUID) -> bool:
        """Check if embedding exists.

        Args:
            embedding_id: Embedding identifier

        Returns:
            True if embedding exists
        """
        embedding = await self.find_by_id(embedding_id)
        return embedding is not None

    async def delete(self, embedding_id: UUID) -> bool:
        """Delete embedding by ID.

        Args:
            embedding_id: Embedding identifier

        Returns:
            True if embedding was deleted, False if not found

        Raises:
            EmbeddingStorageError: If deletion fails
        """
        try:
            return await self._client.delete_object(
                class_name=self.CLASS_NAME,
                uuid=str(embedding_id),
            )

        except Exception as e:
            raise EmbeddingStorageError(
                message=f"Failed to delete embedding {embedding_id}",
                operation="delete",
                embedding_id=embedding_id,
                details=str(e),
            ) from e

    async def delete_by_chunk_id(self, chunk_id: UUID) -> int:
        """Delete embedding by chunk ID.

        Args:
            chunk_id: Chunk identifier

        Returns:
            Number of embeddings deleted

        Raises:
            EmbeddingStorageError: If deletion fails
        """
        embeddings = await self.find_by_chunk_ids([chunk_id])
        deleted = 0

        for embedding in embeddings:
            if await self.delete(embedding.id):
                deleted += 1

        return deleted

    async def delete_by_document_id(self, document_id: UUID) -> int:
        """Delete all embeddings for a document.

        Args:
            document_id: Document identifier

        Returns:
            Number of embeddings deleted

        Raises:
            EmbeddingStorageError: If deletion fails
        """
        embeddings = await self.find_by_document_id(document_id)
        deleted = 0

        for embedding in embeddings:
            if await self.delete(embedding.id):
                deleted += 1

        return deleted

    async def count(self) -> int:
        """Count total number of embeddings.

        Returns:
            Total embedding count

        Raises:
            EmbeddingStorageError: If count operation fails
        """
        try:
            return await self._client.count_objects(self.CLASS_NAME)

        except Exception as e:
            raise EmbeddingStorageError(
                message="Failed to count embeddings",
                operation="count",
                details=str(e),
            ) from e

    async def count_by_model(self, model_name: str) -> int:
        """Count embeddings by model name.

        Args:
            model_name: Embedding model name

        Returns:
            Number of embeddings for the model

        Raises:
            EmbeddingStorageError: If count operation fails
        """
        # Note: Weaviate doesn't have a native count-with-filter,
        # so we query and count results
        try:
            filters = {
                "path": ["model_name"],
                "operator": "Equal",
                "valueString": model_name,
            }

            results = await self._client.similarity_search(
                class_name=self.CLASS_NAME,
                vector=[0.0] * 768,
                limit=100000,  # Large limit
                filters=filters,
                min_certainty=0.0,
                include_vector=False,
            )

            return len(results)

        except Exception as e:
            raise EmbeddingStorageError(
                message=f"Failed to count embeddings for model {model_name}",
                operation="count_by_model",
                details=str(e),
            ) from e

    async def health_check(self) -> bool:
        """Check repository health and connectivity.

        Returns:
            True if repository is healthy and accessible
        """
        try:
            health = await self._client.health_check()
            return health.get("status") == "healthy"

        except Exception:
            return False

    def _embedding_to_properties(self, embedding: Embedding) -> Dict[str, any]:
        """Convert embedding to Weaviate properties.

        Args:
            embedding: Embedding entity

        Returns:
            Properties dictionary
        """
        return {
            "chunk_id": str(embedding.chunk_id) if embedding.chunk_id else None,
            "document_id": (
                str(embedding.document_id) if embedding.document_id else None
            ),
            "sequence": embedding.metadata.get("sequence", 0),
            "model_name": embedding.model_name,
            "model_version": embedding.model_version,
            "doc_type": embedding.metadata.get("doc_type", "unknown"),
            "source_type": embedding.source_type,
            "file_name": embedding.metadata.get("file_name", ""),
            "section": embedding.metadata.get("section", ""),
            "token_count": embedding.metadata.get("token_count", 0),
            "text": embedding.metadata.get("text", ""),
            "created_at": embedding.created_at.isoformat(),
        }

    def _object_to_embedding(
        self, obj: Dict[str, any], embedding_id: UUID
    ) -> Embedding:
        """Convert Weaviate object to embedding.

        Args:
            obj: Weaviate object
            embedding_id: Embedding ID

        Returns:
            Embedding entity
        """
        properties = obj.get("properties", {})
        vector_data = obj.get("vector", [])

        vector = EmbeddingVector.from_list(vector_data) if vector_data else None

        metadata = {
            "sequence": properties.get("sequence", 0),
            "doc_type": properties.get("doc_type", "unknown"),
            "source_type": properties.get("source_type", "chunk"),
            "file_name": properties.get("file_name", ""),
            "section": properties.get("section", ""),
            "token_count": properties.get("token_count", 0),
            "text": properties.get("text", ""),
        }

        return Embedding(
            id=embedding_id,
            vector=vector,
            model_name=properties.get("model_name", ""),
            model_version=properties.get("model_version", "1.0"),
            dimension=len(vector_data) if vector_data else 0,
            chunk_id=(
                UUID(properties["chunk_id"]) if properties.get("chunk_id") else None
            ),
            document_id=(
                UUID(properties["document_id"])
                if properties.get("document_id")
                else None
            ),
            metadata=metadata,
            created_at=datetime.fromisoformat(properties.get("created_at")),
        )

    def _build_weaviate_filters(
        self, filters: Dict[str, any]
    ) -> Dict[str, any]:
        """Build Weaviate filter from simple dict.

        Args:
            filters: Simple filter dictionary

        Returns:
            Weaviate-compatible filter
        """
        # Simple implementation - extend as needed
        operands = []
        for key, value in filters.items():
            operands.append({
                "path": [key],
                "operator": "Equal",
                "valueString": str(value) if not isinstance(value, (int, float)) else None,
                "valueInt": value if isinstance(value, int) else None,
                "valueNumber": value if isinstance(value, float) else None,
            })

        if len(operands) == 1:
            return operands[0]
        else:
            return {"operator": "And", "operands": operands}
