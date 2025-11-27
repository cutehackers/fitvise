"""Embedding domain service (Epic 2.2).

This module defines the domain service coordinating embedding generation,
storage, and retrieval operations.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence
from uuid import UUID

from app.domain.entities.embedding import Embedding
from app.domain.entities.chunk import Chunk
from app.domain.exceptions.embedding_exceptions import (
    DimensionMismatchError,
    EmbeddingGenerationError,
    EmbeddingStorageError,
)
from app.domain.repositories.embedding_repository import EmbeddingRepository
from app.domain.value_objects.embedding_vector import EmbeddingVector
from app.domain.value_objects.embedding_result import EmbeddingResult


class EmbeddingService:
    """Domain service for embedding operations (Task 2.2.1).

    Coordinates between embedding generation, validation, and storage,
    implementing business logic for batch vs real-time processing.

    Examples:
        >>> service = EmbeddingService(repository, sentence_transformer_service)
        >>> # Store embedding
        >>> await service.store_chunk_embedding(
        ...     chunk=chunk,
        ...     vector=vector,
        ...     model_name="Alibaba-NLP/gte-multilingual-base"
        ... )

        >>> # Batch storage
        >>> await service.store_chunk_embeddings_batch(
        ...     chunks=chunks,
        ...     vectors=vectors,
        ...     model_name="Alibaba-NLP/gte-multilingual-base"
        ... )

        >>> # Embed query with caching
        >>> result = await service.embed_query(
        ...     query="What exercises help with back pain?",
        ...     use_cache=True
        ... )
        >>>
        >>> # Search similar
        >>> results = await service.find_similar(
        ...     query_vector=result.vector,
        ...     k=10
        ... )
    """

    def __init__(
        self,
        repository: EmbeddingRepository,
        sentence_transformer_service=None,
    ) -> None:
        """Initialize embedding service.

        Args:
            repository: Embedding repository for persistence
            sentence_transformer_service: Service for generating embeddings
        """
        self._repository = repository
        self._sentence_transformer_service = sentence_transformer_service

    async def store_chunk_embedding(
        self,
        chunk: Chunk,
        vector: EmbeddingVector,
        model_name: str,
        model_version: str = "1.0",
        metadata: Optional[Dict[str, any]] = None,
    ) -> Embedding:
        """Store embedding for a chunk.

        Args:
            chunk: Source chunk
            vector: Embedding vector
            model_name: Name of embedding model
            model_version: Model version
            metadata: Additional metadata

        Returns:
            Stored embedding

        Raises:
            EmbeddingStorageError: If storage fails
        """
        embedding = Embedding.for_chunk(
            vector=vector,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            model_name=model_name,
            model_version=model_version,
            metadata=metadata,
        )

        try:
            await self._repository.save(embedding)
            return embedding
        except Exception as e:
            raise EmbeddingStorageError(
                message="Failed to store chunk embedding",
                operation="save",
                embedding_id=embedding.id,
                details=str(e),
            ) from e

    async def store_chunk_embeddings_batch(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[EmbeddingVector],
        model_name: str,
        model_version: str = "1.0",
        batch_size: int = 100,
    ) -> List[Embedding]:
        """Store embeddings for multiple chunks in batches.

        Args:
            chunks: Source chunks
            vectors: Embedding vectors (same order as chunks)
            model_name: Name of embedding model
            model_version: Model version
            batch_size: Number of embeddings per batch

        Returns:
            List of stored embeddings

        Raises:
            ValueError: If chunks and vectors have different lengths
            EmbeddingStorageError: If batch storage fails
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Length mismatch: {len(chunks)} chunks != {len(vectors)} vectors"
            )

        embeddings = [
            Embedding.for_chunk(
                vector=vector,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                model_name=model_name,
                model_version=model_version,
            )
            for chunk, vector in zip(chunks, vectors)
        ]

        try:
            await self._repository.batch_save(embeddings, batch_size=batch_size)
            return embeddings
        except Exception as e:
            raise EmbeddingStorageError(
                message="Failed to batch store embeddings",
                operation="batch_save",
                details=str(e),
            ) from e

    async def find_by_chunk(self, chunk_id: UUID) -> Optional[Embedding]:
        """Find embedding by chunk ID.

        Args:
            chunk_id: Chunk identifier

        Returns:
            Embedding if found, None otherwise
        """
        return await self._repository.find_by_chunk_id(chunk_id)

    async def find_by_document(self, document_id: UUID) -> List[Embedding]:
        """Find all embeddings for a document.

        Args:
            document_id: Document identifier

        Returns:
            List of embeddings for the document
        """
        return await self._repository.find_by_document_id(document_id)

    async def find_similar(
        self,
        query_vector: EmbeddingVector,
        k: int = 10,
        filters: Optional[Dict[str, any]] = None,
        min_similarity: float = 0.0,
    ) -> List[tuple[Embedding, float]]:
        """Find similar embeddings using vector search.

        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            filters: Optional metadata filters
            min_similarity: Minimum similarity threshold

        Returns:
            List of (embedding, similarity_score) tuples
        """
        return await self._repository.similarity_search(
            query_vector=query_vector,
            k=k,
            filters=filters,
            min_similarity=min_similarity,
        )

    async def validate_dimension_compatibility(
        self,
        embedding: Embedding,
        expected_dimension: int,
    ) -> None:
        """Validate embedding dimension matches expected.

        Args:
            embedding: Embedding to validate
            expected_dimension: Expected vector dimension

        Raises:
            DimensionMismatchError: If dimensions don't match
        """
        if embedding.dimension != expected_dimension:
            raise DimensionMismatchError(
                expected=expected_dimension,
                actual=embedding.dimension,
                details=f"Embedding {embedding.id} has incompatible dimension",
            )

    async def delete_by_chunk(self, chunk_id: UUID) -> int:
        """Delete embedding by chunk ID.

        Args:
            chunk_id: Chunk identifier

        Returns:
            Number of embeddings deleted
        """
        return await self._repository.delete_by_chunk_id(chunk_id)

    async def delete_by_document(self, document_id: UUID) -> int:
        """Delete all embeddings for a document.

        Args:
            document_id: Document identifier

        Returns:
            Number of embeddings deleted
        """
        return await self._repository.delete_by_document_id(document_id)

    async def embed_query(
        self,
        query: str,
        use_cache: bool = True,
        store_embedding: bool = False,
        model_name: str = "Alibaba-NLP/gte-multilingual-base",
        model_version: str = "1.0",
    ) -> EmbeddingResult:
        """Embed a query text with caching support.

        Args:
            query: Query text to embed
            use_cache: Whether to use query cache
            store_embedding: Whether to store the embedding
            model_name: Name of embedding model to use
            model_version: Version of embedding model

        Returns:
            EmbeddingResult with generated vector and metadata

        Raises:
            EmbeddingGenerationError: If embedding generation fails
            EmbeddingStorageError: If storage fails
        """
        try:
            # Validate input
            if not query or not query.strip():
                raise EmbeddingGenerationError(
                    message="Query cannot be empty",
                    operation="embed_query",
                    details="Query validation failed"
                )

            # Check cache first if enabled
            if use_cache:
                cached_embedding = await self._repository.find_by_query_text(query.strip())
                if cached_embedding:
                    return EmbeddingResult.from_cache_hit(
                        query=query.strip(),
                        vector=cached_embedding.vector,
                        model_name=model_name,
                        embedding_id=cached_embedding.id,
                        model_version=model_version,
                        metadata={"cache_hit": True, "model_name": model_name},
                    )

            # Generate embedding using sentence transformer service
            if not self._sentence_transformer_service:
                raise EmbeddingGenerationError(
                    message="Sentence transformer service not available",
                    operation="embed_query",
                    details="EmbeddingService initialized without sentence_transformer_service"
                )

            vector = await self._sentence_transformer_service.encode(query.strip())

            embedding_id = None
            if store_embedding:
                # Store the embedding for future use
                embedding = Embedding.for_query_text(
                    vector=vector,
                    query_text=query.strip(),
                    model_name=model_name,
                    model_version=model_version,
                    metadata={"query_type": "user_query"},
                )
                await self._repository.save(embedding)
                embedding_id = embedding.id

            return EmbeddingResult.create(
                query=query.strip(),
                vector=vector,
                model_name=model_name,
                model_version=model_version,
                cache_hit=False,
                stored=store_embedding,
                embedding_id=embedding_id,
                metadata={"cache_hit": False, "model_name": model_name},
            )

        except EmbeddingGenerationError:
            raise  # Re-raise embedding generation errors
        except EmbeddingStorageError:
            raise  # Re-raise storage errors
        except Exception as e:
            raise EmbeddingGenerationError(
                message="Failed to embed query",
                operation="embed_query",
                details=str(e),
            ) from e

    async def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        model_name: str = "Alibaba-NLP/gte-multilingual-base",
        model_version: str = "1.0",
        store_embeddings: bool = False,
    ) -> List[EmbeddingResult]:
        """Embed multiple texts efficiently in batches.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch
            model_name: Name of embedding model to use
            model_version: Version of embedding model
            store_embeddings: Whether to store the embeddings

        Returns:
            List of EmbeddingResult objects

        Raises:
            EmbeddingGenerationError: If embedding generation fails
            EmbeddingStorageError: If storage fails
        """
        if not texts:
            return []

        try:
            # Validate input
            if not self._sentence_transformer_service:
                raise EmbeddingGenerationError(
                    message="Sentence transformer service not available",
                    operation="embed_batch",
                    details="EmbeddingService initialized without sentence_transformer_service"
                )

            # Filter out empty texts
            valid_texts = [text.strip() for text in texts if text and text.strip()]
            if not valid_texts:
                return []

            # Generate embeddings in batches
            all_vectors = []
            for i in range(0, len(valid_texts), batch_size):
                batch_texts = valid_texts[i:i + batch_size]
                batch_vectors = await self._sentence_transformer_service.encode_batch(batch_texts)
                all_vectors.extend(batch_vectors)

            # Store embeddings if requested
            embedding_ids = [None] * len(valid_texts)
            if store_embeddings:
                embeddings = [
                    Embedding.for_query_text(
                        vector=vector,
                        query_text=text,
                        model_name=model_name,
                        model_version=model_version,
                        metadata={"batch_index": i, "query_type": "batch_query"},
                    )
                    for i, (text, vector) in enumerate(zip(valid_texts, all_vectors))
                ]
                stored_embeddings = await self._repository.batch_save(embeddings, batch_size=batch_size)
                embedding_ids = [emb.id for emb in stored_embeddings]

            # Create results
            results = [
                EmbeddingResult.create(
                    query=text,
                    vector=vector,
                    model_name=model_name,
                    model_version=model_version,
                    cache_hit=False,
                    stored=store_embeddings,
                    embedding_id=embedding_id,
                    metadata={"batch_index": i, "cache_hit": False},
                )
                for i, (text, vector, embedding_id) in enumerate(zip(valid_texts, all_vectors, embedding_ids))
            ]

            return results

        except EmbeddingGenerationError:
            raise  # Re-raise embedding generation errors
        except EmbeddingStorageError:
            raise  # Re-raise storage errors
        except Exception as e:
            raise EmbeddingGenerationError(
                message="Failed to embed batch",
                operation="embed_batch",
                details=str(e),
            ) from e

    async def health_check(self) -> bool:
        """Check embedding service health.

        Returns:
            True if service is healthy
        """
        repository_healthy = await self._repository.health_check()

        # Check sentence transformer service if available
        transformer_healthy = True
        if self._sentence_transformer_service:
            try:
                # Simple health check - try to encode a short text
                await self._sentence_transformer_service.encode("health check")
            except Exception:
                transformer_healthy = False

        return repository_healthy and transformer_healthy
