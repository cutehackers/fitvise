"""Setup embedding infrastructure use case (Task 2.2.1).

This use case initializes the embedding model and Weaviate vector database,
creating necessary schemas and verifying connectivity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.config.ml_models.embedding_model_configs import EmbeddingModelConfig
from app.config.vector_stores.weaviate_config import WeaviateConfig
from app.domain.exceptions.embedding_exceptions import (
    EmbeddingStorageError,
    ModelLoadError,
)
from app.infrastructure.external_services.ml_services.embedding_models.sentence_transformer_service import (
    SentenceTransformerService,
)
from app.infrastructure.external_services.vector_stores.weaviate_client import (
    WeaviateClient,
)
from app.infrastructure.external_services.vector_stores.weaviate_schema import (
    WeaviateSchema,
)


@dataclass
class SetupRequest:
    """Setup embedding infrastructure request."""

    embedding_config: Optional[Dict[str, Any]] = None
    weaviate_config: Optional[Dict[str, Any]] = None
    vector_dimension: int = 384
    recreate_schema: bool = False


@dataclass
class SetupResponse:
    """Setup embedding infrastructure response."""

    success: bool
    embedding_service_status: Dict[str, Any]
    weaviate_status: Dict[str, Any]
    schema_created: bool
    errors: List[str]

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "embedding_service": self.embedding_service_status,
            "weaviate": self.weaviate_status,
            "schema_created": self.schema_created,
            "errors": self.errors,
        }


class SetupEmbeddingInfrastructureUseCase:
    """Use case for setting up embedding infrastructure (Task 2.2.1).

    Initializes Sentence-Transformers model and Weaviate database,
    creates necessary schemas, and verifies system health.

    Examples:
        >>> use_case = SetupEmbeddingInfrastructureUseCase()
        >>> request = SetupRequest()
        >>> response = await use_case.execute(request)
        >>> response.success
        True
        >>> response.embedding_service_status["is_loaded"]
        True
        >>> response.weaviate_status["connected"]
        True
    """

    def __init__(
        self,
        embedding_service: Optional[SentenceTransformerService] = None,
        weaviate_client: Optional[WeaviateClient] = None,
    ) -> None:
        """Initialize setup use case.

        Args:
            embedding_service: Optional pre-configured embedding service
            weaviate_client: Optional pre-configured Weaviate client
        """
        self._embedding_service = embedding_service
        self._weaviate_client = weaviate_client

    async def execute(self, request: SetupRequest) -> SetupResponse:
        """Execute infrastructure setup.

        Args:
            request: Setup request with configuration

        Returns:
            Setup response with status information
        """
        errors = []
        embedding_status = {}
        weaviate_status = {}
        schema_created = False

        # Step 1: Initialize embedding service
        try:
            embedding_status = await self._setup_embedding_service(request)
        except Exception as e:
            errors.append(f"Embedding service setup failed: {str(e)}")
            embedding_status = {"error": str(e)}

        # Step 2: Initialize Weaviate
        try:
            weaviate_status = await self._setup_weaviate(request)
        except Exception as e:
            errors.append(f"Weaviate setup failed: {str(e)}")
            weaviate_status = {"error": str(e)}

        # Step 3: Create schema
        try:
            schema_created = await self._setup_schema(
                request.vector_dimension,
                request.recreate_schema,
            )
        except Exception as e:
            errors.append(f"Schema setup failed: {str(e)}")

        success = len(errors) == 0

        return SetupResponse(
            success=success,
            embedding_service_status=embedding_status,
            weaviate_status=weaviate_status,
            schema_created=schema_created,
            errors=errors,
        )

    async def _setup_embedding_service(
        self, request: SetupRequest
    ) -> Dict[str, Any]:
        """Setup embedding service.

        Args:
            request: Setup request

        Returns:
            Service status dictionary

        Raises:
            ModelLoadError: If model loading fails
        """
        # Create or use existing service
        if self._embedding_service is None:
            config_dict = request.embedding_config or {}
            config = EmbeddingModelConfig.from_dict(config_dict) if config_dict else EmbeddingModelConfig.default()
            self._embedding_service = SentenceTransformerService(config)

        # Initialize model
        await self._embedding_service.initialize()

        # Health check
        health = await self._embedding_service.health_check()

        return {
            "is_loaded": self._embedding_service.is_loaded,
            "model_name": self._embedding_service.model_name,
            "model_dimension": self._embedding_service.model_dimension,
            "device": self._embedding_service._device,
            "health": health,
        }

    async def _setup_weaviate(self, request: SetupRequest) -> Dict[str, Any]:
        """Setup Weaviate connection.

        Args:
            request: Setup request

        Returns:
            Weaviate status dictionary

        Raises:
            EmbeddingStorageError: If connection fails
        """
        # Create or use existing client
        if self._weaviate_client is None:
            config_dict = request.weaviate_config or {}
            config = WeaviateConfig.from_dict(config_dict) if config_dict else WeaviateConfig.for_local_development()
            self._weaviate_client = WeaviateClient(config)

        # Connect
        await self._weaviate_client.connect()

        # Health check
        health = await self._weaviate_client.health_check()

        return {
            "connected": self._weaviate_client.is_connected,
            "url": self._weaviate_client.config.get_url(),
            "health": health,
        }

    async def _setup_schema(
        self, dimension: int, recreate: bool
    ) -> bool:
        """Setup Weaviate schema.

        Args:
            dimension: Embedding vector dimension
            recreate: Whether to recreate schema if exists

        Returns:
            True if schema was created, False if already exists

        Raises:
            EmbeddingStorageError: If schema creation fails
        """
        if self._weaviate_client is None:
            raise EmbeddingStorageError(
                message="Weaviate client not initialized",
                operation="setup_schema",
            )

        schema = WeaviateSchema(self._weaviate_client._client)

        # Check if class exists
        exists = await schema.class_exists("Chunk")

        if exists and recreate:
            # Delete and recreate
            await schema.delete_class("Chunk")
            await schema.create_chunk_class(dimension=dimension)
            return True
        elif not exists:
            # Create new
            await schema.create_chunk_class(dimension=dimension)
            return True
        else:
            # Already exists, no action
            return False

    def get_embedding_service(self) -> Optional[SentenceTransformerService]:
        """Get initialized embedding service.

        Returns:
            Embedding service if initialized
        """
        return self._embedding_service

    def get_weaviate_client(self) -> Optional[WeaviateClient]:
        """Get initialized Weaviate client.

        Returns:
            Weaviate client if initialized
        """
        return self._weaviate_client
