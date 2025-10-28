"""Weaviate client for vector database operations (Epic 2.3).

This module implements the Weaviate client with connection management,
CRUD operations, batch upload, and similarity search capabilities.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID

import weaviate
from weaviate.exceptions import WeaviateBaseError

from app.config.vector_stores.weaviate_config import WeaviateConfig
from app.domain.exceptions.embedding_exceptions import (
    EmbeddingStorageError,
    VectorSearchError,
)

logger = logging.getLogger(__name__)


class WeaviateClient:
    """Weaviate vector database client (Task 2.3.1).

    Provides connection management and operations for Weaviate vector database
    with support for CRUD, batch upload, and similarity search.

    Examples:
        >>> config = WeaviateConfig.for_local_development()
        >>> client = WeaviateClient(config)
        >>> await client.connect()
        >>> client.is_connected
        True

        >>> # Create object
        >>> object_id = await client.create_object(
        ...     class_name="Chunk",
        ...     properties={"text": "Hello", "chunk_id": str(uuid4())},
        ...     vector=[0.1, 0.2, 0.3]
        ... )

        >>> # Batch upload
        >>> await client.batch_create_objects(
        ...     class_name="Chunk",
        ...     objects=[...],
        ...     batch_size=100
        ... )

        >>> # Similarity search
        >>> results = await client.similarity_search(
        ...     class_name="Chunk",
        ...     vector=[0.1, 0.2, 0.3],
        ...     limit=10
        ... )
    """

    def __init__(self, config: WeaviateConfig) -> None:
        """Initialize Weaviate client.

        Args:
            config: Weaviate configuration
        """
        self.config = config
        self._client: Optional[weaviate.Client] = None
        self.is_connected = False

    async def connect(self) -> None:
        """Connect to Weaviate server.

        Raises:
            EmbeddingStorageError: If connection fails
        """
        if self.is_connected and self._client is not None:
            logger.info("Already connected to Weaviate")
            return

        try:
            logger.info(f"Connecting to Weaviate at {self.config.get_url()}")

            # Build authentication
            auth_config = None
            if self.config.auth_type.value == "api_key" and self.config.api_key:
                auth_config = weaviate.AuthApiKey(api_key=self.config.api_key)

            # Connect in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self._client = await loop.run_in_executor(
                None,
                lambda: weaviate.Client(
                    url=self.config.get_url(),
                    auth_client_secret=auth_config,
                    timeout_config=(
                        self.config.connection_timeout,
                        self.config.read_timeout,
                    ),
                    additional_headers=self.config.additional_headers,
                ),
            )

            # Test connection
            is_ready = await loop.run_in_executor(
                None, lambda: self._client.is_ready()
            )

            if not is_ready:
                raise EmbeddingStorageError(
                    message="Weaviate server not ready",
                    operation="connect",
                )

            self.is_connected = True
            logger.info("Successfully connected to Weaviate")

        except WeaviateBaseError as e:
            raise EmbeddingStorageError(
                message="Failed to connect to Weaviate",
                operation="connect",
                details=str(e),
            ) from e
        except Exception as e:
            raise EmbeddingStorageError(
                message="Unexpected error connecting to Weaviate",
                operation="connect",
                details=str(e),
            ) from e

    async def disconnect(self) -> None:
        """Disconnect from Weaviate server."""
        if self._client is not None:
            # Weaviate client doesn't need explicit cleanup in v3
            self._client = None
            self.is_connected = False
            logger.info("Disconnected from Weaviate")

    def validate_connected(self) -> None:
        """Validate client is connected.

        Raises:
            EmbeddingStorageError: If not connected
        """
        if not self.is_connected or self._client is None:
            raise EmbeddingStorageError(
                message="Not connected to Weaviate. Call connect() first.",
                operation="validate_connection",
            )

    async def create_object(
        self,
        class_name: str,
        properties: Dict[str, Any],
        vector: Optional[List[float]] = None,
        uuid: Optional[str] = None,
    ) -> str:
        """Create a single object in Weaviate.

        Args:
            class_name: Name of the Weaviate class
            properties: Object properties
            vector: Optional embedding vector
            uuid: Optional UUID for object

        Returns:
            UUID of created object

        Raises:
            EmbeddingStorageError: If creation fails
        """
        self.validate_connected()

        try:
            loop = asyncio.get_event_loop()
            object_uuid = await loop.run_in_executor(
                None,
                lambda: self._client.data_object.create(
                    class_name=class_name,
                    data_object=properties,
                    vector=vector,
                    uuid=uuid,
                ),
            )

            logger.debug(f"Created object in {class_name}: {object_uuid}")
            return object_uuid

        except WeaviateBaseError as e:
            raise EmbeddingStorageError(
                message=f"Failed to create object in {class_name}",
                operation="create_object",
                details=str(e),
            ) from e

    async def batch_create_objects(
        self,
        class_name: str,
        objects: Sequence[Dict[str, Any]],
        vectors: Optional[Sequence[List[float]]] = None,
        uuids: Optional[Sequence[str]] = None,
        batch_size: int = 100,
    ) -> int:
        """Create multiple objects in batches.

        Args:
            class_name: Name of the Weaviate class
            objects: List of object properties
            vectors: Optional embedding vectors (same order as objects)
            uuids: Optional UUIDs (same order as objects)
            batch_size: Number of objects per batch

        Returns:
            Number of objects successfully created

        Raises:
            EmbeddingStorageError: If batch creation fails
        """
        self.validate_connected()

        if not objects:
            return 0

        try:
            logger.info(
                f"Batch creating {len(objects)} objects in {class_name} "
                f"(batch_size={batch_size})"
            )

            created_count = 0
            loop = asyncio.get_event_loop()

            # Process in batches
            for i in range(0, len(objects), batch_size):
                batch_objects = objects[i : i + batch_size]
                batch_vectors = (
                    vectors[i : i + batch_size] if vectors else [None] * len(batch_objects)
                )
                batch_uuids = (
                    uuids[i : i + batch_size] if uuids else [None] * len(batch_objects)
                )

                # Configure batch
                with self._client.batch.configure(batch_size=batch_size) as batch:
                    for obj, vec, uuid in zip(batch_objects, batch_vectors, batch_uuids):
                        batch.add_data_object(
                            class_name=class_name,
                            data_object=obj,
                            vector=vec,
                            uuid=uuid,
                        )

                created_count += len(batch_objects)

            logger.info(f"Successfully created {created_count} objects in {class_name}")
            return created_count

        except WeaviateBaseError as e:
            raise EmbeddingStorageError(
                message=f"Failed to batch create objects in {class_name}",
                operation="batch_create",
                details=str(e),
            ) from e

    async def get_object(
        self,
        class_name: str,
        uuid: str,
        include_vector: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Get object by UUID.

        Args:
            class_name: Name of the Weaviate class
            uuid: Object UUID
            include_vector: Whether to include vector in response

        Returns:
            Object data if found, None otherwise

        Raises:
            EmbeddingStorageError: If retrieval fails
        """
        self.validate_connected()

        try:
            loop = asyncio.get_event_loop()
            obj = await loop.run_in_executor(
                None,
                lambda: self._client.data_object.get_by_id(
                    uuid=uuid,
                    class_name=class_name,
                    with_vector=include_vector,
                ),
            )

            return obj

        except WeaviateBaseError as e:
            if "not found" in str(e).lower():
                return None
            raise EmbeddingStorageError(
                message=f"Failed to get object {uuid} from {class_name}",
                operation="get_object",
                embedding_id=UUID(uuid),
                details=str(e),
            ) from e

    async def delete_object(self, class_name: str, uuid: str) -> bool:
        """Delete object by UUID.

        Args:
            class_name: Name of the Weaviate class
            uuid: Object UUID

        Returns:
            True if deleted, False if not found

        Raises:
            EmbeddingStorageError: If deletion fails
        """
        self.validate_connected()

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._client.data_object.delete(
                    uuid=uuid,
                    class_name=class_name,
                ),
            )

            logger.debug(f"Deleted object from {class_name}: {uuid}")
            return True

        except WeaviateBaseError as e:
            if "not found" in str(e).lower():
                return False
            raise EmbeddingStorageError(
                message=f"Failed to delete object {uuid} from {class_name}",
                operation="delete_object",
                embedding_id=UUID(uuid),
                details=str(e),
            ) from e

    async def similarity_search(
        self,
        class_name: str,
        vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        min_certainty: float = 0.0,
        include_vector: bool = False,
    ) -> List[Dict[str, Any]]:
        """Search for similar objects using vector similarity.

        Args:
            class_name: Name of the Weaviate class
            vector: Query embedding vector
            limit: Maximum number of results
            filters: Optional property filters
            min_certainty: Minimum certainty threshold (0-1)
            include_vector: Whether to include vectors in results

        Returns:
            List of similar objects with metadata

        Raises:
            VectorSearchError: If search fails
        """
        self.validate_connected()

        try:
            logger.debug(
                f"Searching {class_name} with vector (dim={len(vector)}, "
                f"limit={limit}, min_certainty={min_certainty})"
            )

            # Build query
            query = (
                self._client.query.get(class_name)
                .with_near_vector({"vector": vector, "certainty": min_certainty})
                .with_limit(limit)
            )

            # Add filters if provided
            if filters:
                query = query.with_where(filters)

            # Include vector if requested
            if include_vector:
                query = query.with_additional(["vector", "certainty", "distance"])
            else:
                query = query.with_additional(["certainty", "distance"])

            # Execute query in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: query.do())

            # Extract results
            objects = result.get("data", {}).get("Get", {}).get(class_name, [])

            logger.debug(f"Found {len(objects)} similar objects")
            return objects

        except WeaviateBaseError as e:
            raise VectorSearchError(
                message=f"Failed to search {class_name}",
                query_dimension=len(vector),
                details=str(e),
            ) from e

    async def count_objects(self, class_name: str) -> int:
        """Count total objects in a class.

        Args:
            class_name: Name of the Weaviate class

        Returns:
            Number of objects

        Raises:
            EmbeddingStorageError: If count fails
        """
        self.validate_connected()

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._client.query.aggregate(class_name).with_meta_count().do(),
            )

            count = (
                result.get("data", {})
                .get("Aggregate", {})
                .get(class_name, [{}])[0]
                .get("meta", {})
                .get("count", 0)
            )

            return count

        except WeaviateBaseError as e:
            raise EmbeddingStorageError(
                message=f"Failed to count objects in {class_name}",
                operation="count",
                details=str(e),
            ) from e

    async def health_check(self) -> Dict[str, Any]:
        """Check Weaviate health.

        Returns:
            Health status dictionary
        """
        health = {
            "connected": self.is_connected,
            "url": self.config.get_url(),
        }

        if not self.is_connected or self._client is None:
            health["status"] = "disconnected"
            return health

        try:
            loop = asyncio.get_event_loop()
            is_ready = await loop.run_in_executor(
                None, lambda: self._client.is_ready()
            )
            is_live = await loop.run_in_executor(
                None, lambda: self._client.is_live()
            )

            health["status"] = "healthy" if (is_ready and is_live) else "unhealthy"
            health["ready"] = is_ready
            health["live"] = is_live

        except Exception as e:
            health["status"] = "error"
            health["error"] = str(e)

        return health
