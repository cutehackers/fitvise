"""Weaviate schema definition for embeddings (Task 2.3.2).

This module defines the Weaviate schema for storing document chunks
with embeddings and metadata, optimized for hybrid search.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import weaviate
from weaviate.exceptions import WeaviateBaseError

from app.domain.exceptions.embedding_exceptions import EmbeddingStorageError

logger = logging.getLogger(__name__)


class WeaviateSchema:
    """Weaviate schema manager (Task 2.3.2).

    Manages schema creation, updates, and validation for Weaviate classes.

    Examples:
        >>> schema = WeaviateSchema(client)
        >>> await schema.create_chunk_class(dimension=384)
        >>> exists = await schema.class_exists("Chunk")
        >>> exists
        True

        >>> # Get class info
        >>> class_def = await schema.get_class("Chunk")
        >>> class_def["vectorizer"]
        'none'
    """

    def __init__(self, client: weaviate.Client) -> None:
        """Initialize schema manager.

        Args:
            client: Weaviate client instance
        """
        self._client = client

    async def create_chunk_class(
        self,
        dimension: int = 384,
        distance_metric: str = "cosine",
        description: str = "Document chunks with embeddings for RAG system",
    ) -> None:
        """Create Chunk class for storing embedded chunks.

        Args:
            dimension: Embedding vector dimension
            distance_metric: Distance metric (cosine, dot, l2-squared)
            description: Class description

        Raises:
            EmbeddingStorageError: If class creation fails
        """
        class_schema = create_chunk_class_schema(
            dimension=dimension,
            distance_metric=distance_metric,
            description=description,
        )

        try:
            # Check if class already exists
            if await self.class_exists("Chunk"):
                logger.info("Chunk class already exists, skipping creation")
                return

            # Create class
            self._client.schema.create_class(class_schema)
            logger.info(f"Created Chunk class (dimension={dimension})")

        except WeaviateBaseError as e:
            raise EmbeddingStorageError(
                message="Failed to create Chunk class",
                operation="create_schema",
                details=str(e),
            ) from e

    async def class_exists(self, class_name: str) -> bool:
        """Check if class exists in schema.

        Args:
            class_name: Name of the class

        Returns:
            True if class exists
        """
        try:
            schema = self._client.schema.get()
            classes = schema.get("classes", [])
            return any(cls["class"] == class_name for cls in classes)

        except WeaviateBaseError as e:
            logger.error(f"Failed to check if class exists: {e}")
            return False

    async def get_class(self, class_name: str) -> Optional[Dict[str, Any]]:
        """Get class definition from schema.

        Args:
            class_name: Name of the class

        Returns:
            Class definition if found, None otherwise
        """
        try:
            return self._client.schema.get(class_name)

        except WeaviateBaseError as e:
            if "not found" in str(e).lower():
                return None
            raise EmbeddingStorageError(
                message=f"Failed to get class {class_name}",
                operation="get_schema",
                details=str(e),
            ) from e

    async def delete_class(self, class_name: str) -> bool:
        """Delete class from schema.

        Args:
            class_name: Name of the class

        Returns:
            True if deleted, False if not found

        Raises:
            EmbeddingStorageError: If deletion fails
        """
        try:
            if not await self.class_exists(class_name):
                logger.warning(f"Class {class_name} does not exist")
                return False

            self._client.schema.delete_class(class_name)
            logger.info(f"Deleted class {class_name}")
            return True

        except WeaviateBaseError as e:
            raise EmbeddingStorageError(
                message=f"Failed to delete class {class_name}",
                operation="delete_schema",
                details=str(e),
            ) from e

    async def get_all_classes(self) -> List[str]:
        """Get all class names in schema.

        Returns:
            List of class names
        """
        try:
            schema = self._client.schema.get()
            classes = schema.get("classes", [])
            return [cls["class"] for cls in classes]

        except WeaviateBaseError as e:
            logger.error(f"Failed to get classes: {e}")
            return []


def create_chunk_class_schema(
    dimension: int = 384,
    distance_metric: str = "cosine",
    description: str = "Document chunks with embeddings for RAG system",
) -> Dict[str, Any]:
    """Create Chunk class schema definition (Task 2.3.2).

    Defines schema for storing document chunks with embeddings and metadata.
    Optimized for hybrid search with rich metadata support.

    Schema Properties:
        - text (text): Chunk text content
        - chunk_id (uuid): Unique chunk identifier
        - document_id (uuid): Source document identifier
        - sequence (int): Chunk sequence in document
        - model_name (string): Embedding model name
        - model_version (string): Embedding model version
        - doc_type (string): Document type
        - source_type (string): Source type (chunk/query)
        - file_name (string): Source file name
        - section (string): Document section
        - token_count (int): Token count
        - created_at (date): Creation timestamp

    Vector Configuration:
        - Vectorizer: none (we provide embeddings)
        - Index: HNSW for performance
        - Distance: cosine similarity

    Args:
        dimension: Embedding vector dimension
        distance_metric: Distance metric (cosine, dot, l2-squared)
        description: Class description

    Returns:
        Weaviate class schema dictionary

    Examples:
        >>> schema = create_chunk_class_schema(dimension=384)
        >>> schema["class"]
        'Chunk'
        >>> len(schema["properties"])
        12
        >>> schema["vectorIndexConfig"]["distance"]
        'cosine'
    """
    return {
        "class": "Chunk",
        "description": description,
        "vectorizer": "none",  # We provide embeddings
        "vectorIndexConfig": {
            "distance": distance_metric,  # cosine, dot, l2-squared
            "ef": -1,  # Dynamic ef
            "efConstruction": 128,  # Build-time ef
            "maxConnections": 64,  # HNSW parameter
            "vectorCacheMaxObjects": 1000000,  # Cache size
        },
        "properties": [
            {
                "name": "text",
                "dataType": ["text"],
                "description": "Chunk text content",
                "indexFilterable": True,
                "indexSearchable": True,
            },
            {
                "name": "chunk_id",
                "dataType": ["uuid"],
                "description": "Unique chunk identifier",
                "indexFilterable": True,
            },
            {
                "name": "document_id",
                "dataType": ["uuid"],
                "description": "Source document identifier",
                "indexFilterable": True,
            },
            {
                "name": "sequence",
                "dataType": ["int"],
                "description": "Chunk sequence in document",
                "indexFilterable": True,
                "indexRangeFilters": True,
            },
            {
                "name": "model_name",
                "dataType": ["string"],
                "description": "Embedding model name",
                "indexFilterable": True,
            },
            {
                "name": "model_version",
                "dataType": ["string"],
                "description": "Embedding model version",
                "indexFilterable": True,
            },
            {
                "name": "doc_type",
                "dataType": ["string"],
                "description": "Document type (pdf, docx, etc.)",
                "indexFilterable": True,
            },
            {
                "name": "source_type",
                "dataType": ["string"],
                "description": "Source type (chunk, query, etc.)",
                "indexFilterable": True,
            },
            {
                "name": "file_name",
                "dataType": ["string"],
                "description": "Source file name",
                "indexFilterable": True,
            },
            {
                "name": "section",
                "dataType": ["string"],
                "description": "Document section",
                "indexFilterable": True,
            },
            {
                "name": "token_count",
                "dataType": ["int"],
                "description": "Token count in chunk",
                "indexFilterable": True,
                "indexRangeFilters": True,
            },
            {
                "name": "created_at",
                "dataType": ["date"],
                "description": "Creation timestamp",
                "indexFilterable": True,
                "indexRangeFilters": True,
            },
        ],
    }
