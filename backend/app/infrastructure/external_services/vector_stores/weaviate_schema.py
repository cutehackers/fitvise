"""Weaviate schema definition for embeddings (Task 2.3.2).

This module defines the Weaviate schema for storing document chunks
with embeddings and metadata, optimized for hybrid search and RAG retrieval.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from uuid import UUID

# Conditional import for type checking and runtime
if TYPE_CHECKING:
    import weaviate
    from weaviate.exceptions import WeaviateBaseError
else:
    try:
        import weaviate
        from weaviate.exceptions import WeaviateBaseError
    except ImportError:
        # Weaviate not installed - schema functions still work
        weaviate = None  # type: ignore
        WeaviateBaseError = Exception  # type: ignore

from app.domain.exceptions.embedding_exceptions import EmbeddingStorageError

logger = logging.getLogger(__name__)


# Schema version for tracking migrations
SCHEMA_VERSION = "2.3.2"


class WeaviateSchema:
    """Weaviate schema manager (Task 2.3.2).

    Manages schema creation, updates, and validation for Weaviate classes.

    Examples:
        >>> schema = WeaviateSchema(client)
        >>> await schema.create_chunk_class(dimension=768)
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
        dimension: int = 768,
        distance_metric: str = "cosine",
        description: str = "Document chunks with embeddings for RAG system",
    ) -> None:
        """Create Chunk class for storing embedded chunks using Weaviate v4 API.

        Args:
            dimension: Embedding vector dimension
            distance_metric: Distance metric (cosine, dot, l2-squared)
            description: Class description

        Raises:
            EmbeddingStorageError: If class creation fails
        """
        try:
            # Check if class already exists
            if await self.class_exists("Chunk"):
                logger.info("Chunk class already exists, skipping creation")
                return

            # Create comprehensive properties for RAG system
            properties = [
                # Core Properties
                Property(name="text", data_type=DataType.TEXT, description="Chunk text content"),
                Property(name="chunk_id", data_type=DataType.UUID, description="Unique chunk identifier"),
                Property(name="document_id", data_type=DataType.UUID, description="Source document identifier"),
                Property(name="sequence", data_type=DataType.INT, description="Chunk sequence in document"),

                # Model Metadata
                Property(name="model_name", data_type=DataType.TEXT, description="Embedding model name"),
                Property(name="model_version", data_type=DataType.TEXT, description="Embedding model version"),

                # Document Metadata
                Property(name="doc_type", data_type=DataType.TEXT, description="Document type (pdf, docx, txt, md, etc.)"),
                Property(name="source_type", data_type=DataType.TEXT, description="Source type (chunk, query, etc.)"),
                Property(name="file_name", data_type=DataType.TEXT, description="Source file name"),
                Property(name="section", data_type=DataType.TEXT, description="Document section or chapter"),
                Property(name="category", data_type=DataType.TEXT, description="Document category for filtering"),
                Property(name="department", data_type=DataType.TEXT, description="Department or team for access control"),
                Property(name="author", data_type=DataType.TEXT, description="Document author or creator"),
                Property(name="language", data_type=DataType.TEXT, description="Content language (en, es, fr, etc.)"),

                # Hierarchical Context
                Property(name="parent_chunk_id", data_type=DataType.UUID, description="Parent chunk for hierarchical retrieval"),
                Property(name="tags", data_type=DataType.TEXT_ARRAY, description="Tags for flexible categorization"),

                # Quality Metrics
                Property(name="token_count", data_type=DataType.INT, description="Token count in chunk"),
                Property(name="confidence_score", data_type=DataType.NUMBER, description="Chunking confidence score (0-1)"),
                Property(name="quality_score", data_type=DataType.NUMBER, description="Content quality score (0-1)"),

                # Timestamps
                Property(name="created_at", data_type=DataType.DATE, description="Creation timestamp"),
                Property(name="updated_at", data_type=DataType.DATE, description="Last update timestamp"),
            ]

            # Create class using v4 API
            self._client.collections.create(
                name="Chunk",
                vectorizer_config=Configure.Vectorizer.none(),
                generative_config=Configure.Generative.none(),
                properties=properties,
            )
            logger.info(f"Created Chunk class with {len(properties)} properties (dimension={dimension})")

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
            # In Weaviate v4, use collections.exists instead of schema.get
            return self._client.collections.exists(class_name)

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
            # In Weaviate v4, collections.config.get returns collection configuration
            return self._client.collections.config.get(class_name)

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

            # In Weaviate v4, use collections.delete instead of schema.delete_class
            self._client.collections.delete(class_name)
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
            # In Weaviate v4, use collections.list_all to get all collections
            collections = self._client.collections.list_all()
            return [collection.name for collection in collections]

        except WeaviateBaseError as e:
            logger.error(f"Failed to get classes: {e}")
            return []


def validate_chunk_metadata(metadata: Dict[str, Any]) -> None:
    """Validate chunk metadata against schema (Task 2.3.2).

    Args:
        metadata: Chunk metadata dictionary

    Raises:
        ValueError: If metadata is invalid
    """
    # Required fields
    required = ["chunk_id", "document_id", "text"]
    for field in required:
        if field not in metadata:
            raise ValueError(f"Missing required field: {field}")

    # Validate UUIDs
    uuid_fields = ["chunk_id", "document_id", "parent_chunk_id"]
    for field in uuid_fields:
        if field in metadata and metadata[field] is not None:
            try:
                if not isinstance(metadata[field], UUID):
                    UUID(str(metadata[field]))
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid UUID for {field}: {metadata[field]}") from e

    # Validate numeric ranges
    if "sequence" in metadata and metadata["sequence"] is not None:
        if not isinstance(metadata["sequence"], int) or metadata["sequence"] < 0:
            raise ValueError(f"sequence must be non-negative int: {metadata['sequence']}")

    if "token_count" in metadata and metadata["token_count"] is not None:
        if not isinstance(metadata["token_count"], int) or metadata["token_count"] < 0:
            raise ValueError(
                f"token_count must be non-negative int: {metadata['token_count']}"
            )

    # Validate score ranges (0-1)
    score_fields = ["confidence_score", "quality_score"]
    for field in score_fields:
        if field in metadata and metadata[field] is not None:
            score = metadata[field]
            if not isinstance(score, (int, float)) or not 0 <= score <= 1:
                raise ValueError(f"{field} must be float in [0,1]: {score}")

    # Validate string fields
    string_fields = [
        "model_name",
        "model_version",
        "doc_type",
        "source_type",
        "file_name",
        "section",
        "category",
        "department",
        "author",
        "language",
    ]
    for field in string_fields:
        if field in metadata and metadata[field] is not None:
            if not isinstance(metadata[field], str):
                raise ValueError(f"{field} must be string: {metadata[field]}")

    # Validate tags array
    if "tags" in metadata and metadata["tags"] is not None:
        if not isinstance(metadata["tags"], list):
            raise ValueError(f"tags must be list: {metadata['tags']}")
        if not all(isinstance(tag, str) for tag in metadata["tags"]):
            raise ValueError(f"All tags must be strings: {metadata['tags']}")

    # Validate dates
    date_fields = ["created_at", "updated_at"]
    for field in date_fields:
        if field in metadata and metadata[field] is not None:
            if not isinstance(metadata[field], (datetime, str)):
                raise ValueError(f"{field} must be datetime or ISO string: {metadata[field]}")


def create_chunk_metadata_template(
    chunk_id: UUID,
    document_id: UUID,
    text: str,
    sequence: int = 0,
    model_name: str = "Alibaba-NLP/gte-multilingual-base",
    model_version: str = "1.0",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Create metadata template for chunk insertion (Task 2.3.2).

    Args:
        chunk_id: Unique chunk identifier
        document_id: Source document identifier
        text: Chunk text content
        sequence: Chunk sequence in document
        model_name: Embedding model name
        model_version: Embedding model version
        **kwargs: Additional metadata fields

    Returns:
        Metadata dictionary ready for Weaviate insertion

    Examples:
        >>> from uuid import uuid4
        >>> meta = create_chunk_metadata_template(
        ...     chunk_id=uuid4(),
        ...     document_id=uuid4(),
        ...     text="Sample chunk text",
        ...     sequence=0,
        ...     doc_type="pdf",
        ...     department="engineering"
        ... )
        >>> "chunk_id" in meta
        True
        >>> "created_at" in meta
        True
    """
    now = datetime.utcnow()

    metadata = {
        # Core properties
        "chunk_id": chunk_id,
        "document_id": document_id,
        "text": text,
        "sequence": sequence,
        # Model metadata
        "model_name": model_name,
        "model_version": model_version,
        # Document metadata (optional)
        "doc_type": kwargs.get("doc_type"),
        "source_type": kwargs.get("source_type", "chunk"),
        "file_name": kwargs.get("file_name"),
        "section": kwargs.get("section"),
        "category": kwargs.get("category"),
        "department": kwargs.get("department"),
        "author": kwargs.get("author"),
        "language": kwargs.get("language", "en"),
        # Hierarchical context
        "parent_chunk_id": kwargs.get("parent_chunk_id"),
        "tags": kwargs.get("tags", []),
        # Quality metrics
        "token_count": kwargs.get("token_count", len(text.split())),
        "confidence_score": kwargs.get("confidence_score", 1.0),
        "quality_score": kwargs.get("quality_score", 1.0),
        # Timestamps
        "created_at": kwargs.get("created_at", now),
        "updated_at": kwargs.get("updated_at", now),
    }

    # Remove None values for optional fields
    return {k: v for k, v in metadata.items() if v is not None}


def get_filterable_fields() -> List[str]:
    """Get list of filterable metadata fields (Task 2.3.2).

    Returns:
        List of field names that support filtering
    """
    return [
        "chunk_id",
        "document_id",
        "sequence",
        "model_name",
        "model_version",
        "doc_type",
        "source_type",
        "file_name",
        "section",
        "category",
        "department",
        "author",
        "language",
        "parent_chunk_id",
        "tags",
        "token_count",
        "confidence_score",
        "quality_score",
        "created_at",
        "updated_at",
    ]


def get_searchable_fields() -> List[str]:
    """Get list of searchable metadata fields (Task 2.3.2).

    Returns:
        List of field names that support text search
    """
    return ["text"]
