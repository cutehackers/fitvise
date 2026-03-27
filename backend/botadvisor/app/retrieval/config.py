"""Canonical retrieval configuration for BotAdvisor."""

from __future__ import annotations

from dataclasses import dataclass

from llama_index.core.vector_stores.types import VectorStoreQueryMode


@dataclass(frozen=True)
class RetrievalConfig:
    """Small retrieval config focused on the canonical hybrid Weaviate path."""

    index_name: str = "BotAdvisorDocs"
    text_key: str = "text"
    embed_model_name: str = "Alibaba-NLP/gte-multilingual-base"
    similarity_top_k: int = 5
    query_mode: VectorStoreQueryMode = VectorStoreQueryMode.HYBRID
    alpha: float = 0.5
    sparse_top_k: int | None = None
    hybrid_top_k: int | None = None
