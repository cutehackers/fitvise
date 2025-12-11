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
from dependency_injector.wiring import Provide, inject

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
from app.di.container import FitviseContainer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


# ============================================================================
# Dependency Injection - Now using DI Container
# ============================================================================

# Type aliases for DI providers
EmbeddingServiceProvider = Provide[FitviseContainer.services.sentence_transformer_service]
WeaviateClientProvider = Provide[FitviseContainer.external.weaviate_client]
EmbeddingRepositoryProvider = Provide[FitviseContainer.repositories.embedding_repository]
EmbeddingDomainServiceProvider = Provide[FitviseContainer.services.embedding_domain_service]
EmbedDocumentChunksUseCaseProvider = Provide[FitviseContainer.services.embed_document_chunks_use_case]
EmbedQueryUseCaseProvider = Provide[FitviseContainer.services.embed_query_use_case]
BatchEmbedUseCaseProvider = Provide[FitviseContainer.services.batch_embed_use_case]
SearchEmbeddingsUseCaseProvider = Provide[FitviseContainer.services.search_embeddings_use_case]
SetupEmbeddingInfrastructureUseCaseProvider = Provide[FitviseContainer.services.setup_embedding_infrastructure_use_case]


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
@inject
async def setup_infrastructure(
    request: SetupInfrastructureRequest,
    use_case: SetupEmbeddingInfrastructureUseCase = Depends(SetupEmbeddingInfrastructureUseCaseProvider),
) -> SetupInfrastructureResponse:
    """Setup embedding infrastructure endpoint.

    Initializes:
    - Sentence-Transformers embedding model
    - Weaviate vector database connection
    - Database schema for embeddings

    Args:
        request: Setup configuration request
        use_case: DI-provided setup use case

    Returns:
        Setup status and configuration details

    Raises:
        HTTPException: If setup fails
    """
    try:
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
@inject
async def embed_chunks(
    request: EmbedChunksRequest,
    use_case: EmbedDocumentChunksUseCase = Depends(EmbedDocumentChunksUseCaseProvider),
) -> EmbedChunksResponse:
    """Embed document chunks endpoint.

    Args:
        request: Chunks to embed
        use_case: DI-provided embed chunks use case

    Returns:
        Embedding results with metadata

    Raises:
        HTTPException: If embedding fails
    """
    try:

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
@inject
async def embed_query(
    request: EmbedQueryRequest,
    use_case: EmbedQueryUseCase = Depends(EmbedQueryUseCaseProvider),
) -> EmbedQueryResponse:
    """Embed query endpoint.

    Args:
        request: Query to embed
        use_case: DI-provided embed query use case

    Returns:
        Query embedding with performance metrics

    Raises:
        HTTPException: If embedding fails
    """
    try:

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
@inject
async def batch_embed(
    request: BatchEmbedRequest,
    use_case: BatchEmbedUseCase = Depends(BatchEmbedUseCaseProvider),
) -> BatchEmbedResponse:
    """Batch embed endpoint.

    Args:
        request: Batch of texts to embed
        use_case: DI-provided batch embed use case

    Returns:
        Batch embedding results with throughput metrics

    Raises:
        HTTPException: If batch embedding fails
    """
    try:

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
@inject
async def search_embeddings(
    request: SearchRequest,
    use_case: SearchEmbeddingsUseCase = Depends(SearchEmbeddingsUseCaseProvider),
) -> SearchResponse:
    """Search embeddings endpoint.

    Args:
        request: Search query and parameters
        use_case: DI-provided search embeddings use case

    Returns:
        Search results with similarity scores

    Raises:
        HTTPException: If search fails
    """
    try:

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
@inject
async def health_check(
    embedding_service: SentenceTransformerService = Depends(EmbeddingServiceProvider),
    repository: EmbeddingRepository = Depends(EmbeddingRepositoryProvider),
) -> Dict[str, Any]:
    """Health check endpoint.

    Args:
        embedding_service: DI-provided embedding service
        repository: DI-provided repository

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
