"""Embedding use cases for RAG system (Task 2.2.1).

This module provides all embedding-related use cases for the RAG pipeline:
- SetupEmbeddingInfrastructureUseCase: Initialize embedding model and Weaviate
- EmbedDocumentChunksUseCase: Generate embeddings for document chunks
- EmbedQueryUseCase: Generate embeddings for user queries with caching
- BatchEmbedUseCase: Large-scale batch embedding operations
- SearchEmbeddingsUseCase: Similarity search on stored embeddings
"""

from .batch_embed import BatchEmbedUseCase
from .embed_document_chunks import EmbedDocumentChunksUseCase
from .embed_query import EmbedQueryUseCase
from .search_embeddings import SearchEmbeddingsUseCase
from .setup_embedding_infrastructure import SetupEmbeddingInfrastructureUseCase

__all__ = [
    "SetupEmbeddingInfrastructureUseCase",
    "EmbedDocumentChunksUseCase",
    "EmbedQueryUseCase",
    "BatchEmbedUseCase",
    "SearchEmbeddingsUseCase",
]
