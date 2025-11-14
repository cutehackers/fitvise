"""Weaviate LangChain retriever adapter.

Bridges domain WeaviateSearchRepository with LangChain's BaseRetriever interface
to enable RAG chain integration.
"""

import logging
from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.infrastructure.repositories.weaviate_search_repository import (
    WeaviateSearchRepository,
)

logger = logging.getLogger(__name__)


class WeaviateLangChainRetriever(BaseRetriever):
    """LangChain retriever adapter for Weaviate search repository.

    Wraps WeaviateSearchRepository to implement LangChain's BaseRetriever interface.
    Converts SearchResult entities to LangChain Document objects.

    Args:
        search_repository: WeaviateSearchRepository instance
        top_k: Number of chunks to retrieve (default: 5)
        similarity_threshold: Minimum similarity score (0.0-1.0, default: 0.7)
    """

    search_repository: WeaviateSearchRepository
    top_k: int = 5
    similarity_threshold: float = 0.7

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Synchronous retrieval (not used in async context).

        Args:
            query: User query
            run_manager: Callback manager

        Returns:
            List of LangChain Document objects

        Raises:
            NotImplementedError: Use aget_relevant_documents for async
        """
        raise NotImplementedError(
            "Use aget_relevant_documents for async retrieval with Weaviate"
        )

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Async retrieval from Weaviate.

        Args:
            query: User query
            run_manager: Callback manager (unused but required by interface)

        Returns:
            List of LangChain Document objects with metadata
        """
        try:
            # Execute semantic search via repository
            search_results = await self.search_repository.semantic_search(
                query=query, top_k=self.top_k, min_similarity=self.similarity_threshold
            )

            # Convert SearchResult entities to LangChain Documents
            documents = []
            for result in search_results:
                # Build document metadata
                metadata = {
                    "chunk_id": result.chunk_id,
                    "document_id": result.document_id,
                    "similarity_score": result.similarity_score,
                    "source": result.metadata.get("source", "unknown"),
                }

                # Add all result metadata
                metadata.update(result.metadata)

                # Create LangChain Document
                doc = Document(page_content=result.content, metadata=metadata)
                documents.append(doc)

            logger.debug(
                "Retrieved %d documents for query (threshold=%.2f)",
                len(documents),
                self.similarity_threshold,
            )

            return documents

        except Exception as e:
            logger.error("Retrieval failed: %s", str(e))
            raise
