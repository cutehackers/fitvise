"""LlamaIndex-backed retrieval service for the canonical BotAdvisor runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters

from botadvisor.app.core.entity.chunk import Chunk
from botadvisor.app.core.entity.document_metadata import DocumentMetadata
from botadvisor.app.core.entity.retriever_request import RetrieverRequest
from botadvisor.app.retrieval.config import RetrievalConfig


@dataclass
class LlamaIndexHybridRetrievalService:
    """Canonical hybrid retrieval service with request-level filter support."""

    index: Any
    config: RetrievalConfig

    def retrieve(self, request: RetrieverRequest) -> list[Chunk]:
        """Retrieve chunks for a request using LlamaIndex hybrid search."""
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=request.top_k,
            vector_store_query_mode=self.config.query_mode,
            filters=self._build_filters(request),
            alpha=self.config.alpha,
            sparse_top_k=self.config.sparse_top_k or request.top_k,
            hybrid_top_k=self.config.hybrid_top_k or request.top_k,
        )
        nodes = retriever.retrieve(request.query)
        return [self._node_to_chunk(node) for node in nodes]

    def _build_filters(self, request: RetrieverRequest) -> MetadataFilters | None:
        filters: list[MetadataFilter] = []

        if request.platform:
            filters.append(MetadataFilter(key="platform", value=request.platform))

        for key, value in request.filters.items():
            filters.append(MetadataFilter(key=key, value=value))

        if not filters:
            return None

        return MetadataFilters(filters=filters)

    def _node_to_chunk(self, node_with_score: Any) -> Chunk:
        node = node_with_score.node
        metadata = dict(getattr(node, "metadata", {}) or {})
        content = node.get_content() or metadata.get("text", "")

        chunk_id = (
            metadata.get("chunk_id")
            or getattr(node, "node_id", None)
            or f"{metadata.get('doc_id', 'unknown')}_{metadata.get('section', 'chunk')}"
        )

        return Chunk(
            chunk_id=chunk_id,
            content=content,
            metadata=DocumentMetadata(
                doc_id=metadata.get("doc_id", ""),
                source_id=metadata.get("source_id", ""),
                platform=metadata.get("platform", ""),
                source_url=metadata.get("source_url", ""),
                page=metadata.get("page"),
                section=metadata.get("section"),
            ),
            score=node_with_score.score,
        )
