"""Factory for the canonical retrieval core."""

from __future__ import annotations

import logging
from typing import Any

from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.weaviate import WeaviateVectorStore

from botadvisor.app.retrieval.config import RetrievalConfig
from botadvisor.app.retrieval.service import LlamaIndexHybridRetrievalService


logger = logging.getLogger(__name__)


def create_hybrid_retriever(weaviate_client: Any, config: RetrievalConfig | None = None) -> LlamaIndexHybridRetrievalService:
    """Create the canonical Weaviate-backed hybrid retriever service."""
    config = config or RetrievalConfig()

    if not getattr(weaviate_client, "is_connected", False):
        raise ValueError("Weaviate client must be connected before creating the retriever.")

    vector_store = WeaviateVectorStore(
        weaviate_client=weaviate_client._client,
        index_name=config.index_name,
        text_key=config.text_key,
    )
    embed_model = HuggingFaceEmbedding(
        model_name=config.embed_model_name,
        trust_remote_code=True,
    )
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )
    logger.info("Created hybrid retrieval index for %s", config.index_name)
    return LlamaIndexHybridRetrievalService(index=index, config=config)
