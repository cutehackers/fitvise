"""Embeddings API endpoints (Task 2.2.1).

REST API endpoints for embedding operations:
- POST /setup - Initialize embedding infrastructure
- POST /embed/chunks - Embed document chunks
- POST /embed/query - Embed user query
- POST /embed/batch - Batch embedding operation
- POST /search - Similarity search
- GET /health - Health check
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.v1.embeddings.schemas import (
    BatchEmbedRequest,
    BatchEmbedResponse,
    EmbedChunksRequest,
    EmbedChunksResponse,
    EmbedQueryRequest,
    EmbedQueryResponse,
    SearchRequest,
    SearchResponse,
    SetupInfrastructureRequest,
    SetupInfrastructureResponse,
)
from app.application.use_cases.embedding import (
    BatchEmbedUseCase,
    EmbedDocumentChunksUseCase,
    EmbedQueryUseCase,
    SearchEmbeddingsUseCase,
    SetupEmbeddingInfrastructureUseCase,
)
from app.config.ml_models.embedding_model_configs import EmbeddingModelConfig
from app.config.vector_stores.weaviate_config import WeaviateConfig
from app.domain.entities.chunk import Chunk
from app.domain.repositories.embedding_repository import EmbeddingRepository
from app.domain.services.embedding_service import EmbeddingService
from app.domain.value_objects.embedding_vector import EmbeddingVector
from app.infrastructure.external_services.ml_services.embedding_models.sentence_transformer_service import (
    SentenceTransformerService,
)
from app.infrastructure.external_services.vector_stores.weaviate_client import (
    WeaviateClient,
)
from app.infrastructure.persistence.repositories.weaviate_embedding_repository import (
    WeaviateEmbeddingRepository,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


# ============================================================================
# Dependency Injection
# ============================================================================


async def get_embedding_service() -> SentenceTransformerService:
    """Get embedding service instance.

    Returns:
        Configured embedding service
    """
    config = EmbeddingModelConfig.for_realtime()
    service = SentenceTransformerService(config)
    await service.initialize()
    return service


async def get_weaviate_client() -> WeaviateClient:
    """Get Weaviate client instance.

    Returns:
        Connected Weaviate client
    """
    config = WeaviateConfig.local()
    client = WeaviateClient(config)
    await client.connect()
    return client


async def get_embedding_repository(
    weaviate_client: WeaviateClient = Depends(get_weaviate_client),
) -> EmbeddingRepository:
    """Get embedding repository instance.

    Args:
        weaviate_client: Weaviate client dependency

    Returns:
        Configured embedding repository
    """
    return WeaviateEmbeddingRepository(weaviate_client)


async def get_embedding_domain_service(
    repository: EmbeddingRepository = Depends(get_embedding_repository),
) -> EmbeddingService:
    """Get embedding domain service instance.

    Args:
        repository: Embedding repository dependency

    Returns:
        Configured domain service
    """
    return EmbeddingService(repository)


# ============================================================================
# Endpoints
# ============================================================================


@router.post(
    "/setup",
    response_model=SetupInfrastructureResponse,
    status_code=status.HTTP_200_OK,
    summary="Setup embedding infrastructure",
    description="Initialize embedding model and Weaviate vector database",
)
async def setup_infrastructure(
    request: SetupInfrastructureRequest,
) -> SetupInfrastructureResponse:
    """Setup embedding infrastructure endpoint.

    Initializes:
    - Sentence-Transformers embedding model
    - Weaviate vector database connection
    - Database schema for embeddings

    Args:
        request: Setup configuration request

    Returns:
        Setup status and configuration details

    Raises:
        HTTPException: If setup fails
    """
    try:
        use_case = SetupEmbeddingInfrastructureUseCase()

        from app.application.use_cases.embedding.setup_embedding_infrastructure import (
            SetupRequest,
        )

        setup_request = SetupRequest(
            embedding_config=request.embedding_config,
            weaviate_config=request.weaviate_config,
            vector_dimension=request.vector_dimension,
            recreate_schema=request.recreate_schema,
        )

        response = await use_case.execute(setup_request)

        if not response.success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "message": "Failed to setup embedding infrastructure",
                    "errors": response.errors,
                },
            )

        return SetupInfrastructureResponse(
            success=response.success,
            embedding_service=response.embedding_service_status,
            weaviate=response.weaviate_status,
            schema_created=response.schema_created,
            errors=response.errors,
        )

    except Exception as e:
        logger.error(f"Setup infrastructure failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Setup failed: {str(e)}",
        )


@router.post(
    "/embed/chunks",
    response_model=EmbedChunksResponse,
    status_code=status.HTTP_200_OK,
    summary="Embed document chunks",
    description="Generate embeddings for document chunks and store in Weaviate",
)
async def embed_chunks(
    request: EmbedChunksRequest,
    embedding_service: SentenceTransformerService = Depends(get_embedding_service),
    repository: EmbeddingRepository = Depends(get_embedding_repository),
    domain_service: EmbeddingService = Depends(get_embedding_domain_service),
) -> EmbedChunksResponse:
    """Embed document chunks endpoint.

    Args:
        request: Chunks to embed
        embedding_service: Embedding service dependency
        repository: Repository dependency
        domain_service: Domain service dependency

    Returns:
        Embedding results with metadata

    Raises:
        HTTPException: If embedding fails
    """
    try:
        use_case = EmbedDocumentChunksUseCase(
            embedding_service=embedding_service,
            embedding_repository=repository,
            domain_service=domain_service,
        )

        # Convert API chunks to domain chunks
        from app.application.use_cases.embedding.embed_document_chunks import (
            EmbedChunksRequest as UseCaseRequest,
        )

        chunks = [
            Chunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text=chunk.text,
                sequence=chunk.sequence,
                metadata=chunk.metadata,
            )
            for chunk in request.chunks
        ]

        use_case_request = UseCaseRequest(
            chunks=chunks,
            model_name=request.model_name,
            model_version=request.model_version,
            batch_size=request.batch_size,
            show_progress=request.show_progress,
            store_embeddings=request.store_embeddings,
        )

        response = await use_case.execute(use_case_request)

        from app.api.v1.embeddings.schemas import ChunkEmbeddingResult

        return EmbedChunksResponse(
            success=response.success,
            total_chunks=response.total_chunks,
            embedded_count=response.embedded_count,
            stored_count=response.stored_count,
            results=[
                ChunkEmbeddingResult(
                    chunk_id=r.chunk_id,
                    embedding_id=r.embedding_id,
                    success=r.success,
                    error=r.error,
                )
                for r in response.results
            ],
            errors=response.errors,
        )

    except Exception as e:
        logger.error(f"Embed chunks failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding failed: {str(e)}",
        )


@router.post(
    "/embed/query",
    response_model=EmbedQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Embed user query",
    description="Generate embedding for user query with caching optimization",
)
async def embed_query(
    request: EmbedQueryRequest,
    embedding_service: SentenceTransformerService = Depends(get_embedding_service),
    repository: EmbeddingRepository = Depends(get_embedding_repository),
    domain_service: EmbeddingService = Depends(get_embedding_domain_service),
) -> EmbedQueryResponse:
    """Embed query endpoint.

    Args:
        request: Query to embed
        embedding_service: Embedding service dependency
        repository: Repository dependency
        domain_service: Domain service dependency

    Returns:
        Query embedding with performance metrics

    Raises:
        HTTPException: If embedding fails
    """
    try:
        use_case = EmbedQueryUseCase(
            embedding_service=embedding_service,
            embedding_repository=repository,
            domain_service=domain_service,
        )

        from app.application.use_cases.embedding.embed_query import (
            EmbedQueryRequest as UseCaseRequest,
        )

        use_case_request = UseCaseRequest(
            query=request.query,
            model_name=request.model_name,
            model_version=request.model_version,
            use_cache=request.use_cache,
            store_embedding=request.store_embedding,
            query_metadata=request.query_metadata,
        )

        response = await use_case.execute(use_case_request)

        return EmbedQueryResponse(
            success=response.success,
            query_id=response.query_id,
            embedding_id=response.embedding_id,
            vector_dimension=response.vector_dimension,
            processing_time_ms=response.processing_time_ms,
            cache_hit=response.cache_hit,
            stored=response.stored,
            error=response.error,
        )

    except Exception as e:
        logger.error(f"Embed query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query embedding failed: {str(e)}",
        )


@router.post(
    "/embed/batch",
    response_model=BatchEmbedResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch embed texts",
    description="Generate embeddings for large batches of texts with progress tracking",
)
async def batch_embed(
    request: BatchEmbedRequest,
    embedding_service: SentenceTransformerService = Depends(get_embedding_service),
    repository: EmbeddingRepository = Depends(get_embedding_repository),
    domain_service: EmbeddingService = Depends(get_embedding_domain_service),
) -> BatchEmbedResponse:
    """Batch embed endpoint.

    Args:
        request: Batch of texts to embed
        embedding_service: Embedding service dependency
        repository: Repository dependency
        domain_service: Domain service dependency

    Returns:
        Batch embedding results with throughput metrics

    Raises:
        HTTPException: If batch embedding fails
    """
    try:
        use_case = BatchEmbedUseCase(
            embedding_service=embedding_service,
            embedding_repository=repository,
            domain_service=domain_service,
        )

        from app.application.use_cases.embedding.batch_embed import (
            BatchEmbedRequest as UseCaseRequest,
        )
        from app.application.use_cases.embedding.batch_embed import TextItem

        # Convert API request to use case request
        text_items = None
        if request.text_items:
            text_items = [
                TextItem(
                    text=item.text,
                    item_id=item.item_id,
                    metadata=item.metadata,
                )
                for item in request.text_items
            ]

        use_case_request = UseCaseRequest(
            texts=request.texts,
            text_items=text_items,
            model_name=request.model_name,
            model_version=request.model_version,
            batch_size=request.batch_size,
            show_progress=request.show_progress,
            store_embeddings=request.store_embeddings,
            storage_batch_size=request.storage_batch_size,
        )

        response = await use_case.execute(use_case_request)

        from app.api.v1.embeddings.schemas import BatchEmbedResult

        return BatchEmbedResponse(
            success=response.success,
            total_items=response.total_items,
            embedded_count=response.embedded_count,
            stored_count=response.stored_count,
            failed_count=response.failed_count,
            processing_time_ms=response.processing_time_ms,
            throughput_items_per_second=response.throughput_items_per_second,
            results=[
                BatchEmbedResult(
                    item_id=r.item_id,
                    embedding_id=r.embedding_id,
                    success=r.success,
                    error=r.error,
                )
                for r in response.results
            ],
            errors=response.errors,
        )

    except Exception as e:
        logger.error(f"Batch embed failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch embedding failed: {str(e)}",
        )


@router.post(
    "/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Search embeddings",
    description="Perform similarity search on stored embeddings",
)
async def search_embeddings(
    request: SearchRequest,
    embedding_service: SentenceTransformerService = Depends(get_embedding_service),
    repository: EmbeddingRepository = Depends(get_embedding_repository),
    domain_service: EmbeddingService = Depends(get_embedding_domain_service),
) -> SearchResponse:
    """Search embeddings endpoint.

    Args:
        request: Search query and parameters
        embedding_service: Embedding service dependency
        repository: Repository dependency
        domain_service: Domain service dependency

    Returns:
        Search results with similarity scores

    Raises:
        HTTPException: If search fails
    """
    try:
        use_case = SearchEmbeddingsUseCase(
            embedding_service=embedding_service,
            embedding_repository=repository,
            domain_service=domain_service,
        )

        from app.application.use_cases.embedding.search_embeddings import (
            SearchRequest as UseCaseRequest,
        )

        # Convert query_vector if provided
        query_vector = None
        if request.query_vector:
            query_vector = EmbeddingVector.from_list(request.query_vector)

        use_case_request = UseCaseRequest(
            query=request.query,
            query_vector=query_vector,
            k=request.k,
            min_similarity=request.min_similarity,
            filters=request.filters,
            include_vectors=request.include_vectors,
            model_name=request.model_name,
        )

        response = await use_case.execute(use_case_request)

        from app.api.v1.embeddings.schemas import SearchResultItem

        return SearchResponse(
            success=response.success,
            query=response.query,
            total_results=response.total_results,
            processing_time_ms=response.processing_time_ms,
            results=[
                SearchResultItem(
                    embedding_id=r.embedding.id,
                    chunk_id=r.embedding.chunk_id,
                    document_id=r.embedding.document_id,
                    similarity_score=r.similarity_score,
                    rank=r.rank,
                    metadata=r.embedding.metadata,
                    vector_dimension=(
                        r.embedding.vector.dimension
                        if request.include_vectors and r.embedding.vector
                        else None
                    ),
                )
                for r in response.results
            ],
            error=response.error,
        )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )


@router.get(
    "/health",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Check embedding service and Weaviate health",
)
async def health_check(
    embedding_service: SentenceTransformerService = Depends(get_embedding_service),
    repository: EmbeddingRepository = Depends(get_embedding_repository),
) -> Dict[str, Any]:
    """Health check endpoint.

    Args:
        embedding_service: Embedding service dependency
        repository: Repository dependency

    Returns:
        Health status of services

    Raises:
        HTTPException: If health check fails
    """
    try:
        embedding_health = await embedding_service.health_check()
        repository_health = await repository.health_check()

        return {
            "status": "healthy" if embedding_health and repository_health else "degraded",
            "embedding_service": {
                "healthy": embedding_health,
                "model_loaded": embedding_service.is_loaded,
                "model_name": embedding_service.model_name,
            },
            "weaviate": {
                "healthy": repository_health,
            },
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}",
        )
