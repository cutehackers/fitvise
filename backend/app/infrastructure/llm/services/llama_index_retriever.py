"""LlamaIndex Weaviate retriever with LangChain compatibility.

Uses LlamaIndex's native VectorStoreIndex with Weaviate backend,
wrapped in a flexible wrapper for LangChain BaseRetriever compatibility.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever as LlamaIndexBaseRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from pydantic import Field

from app.infrastructure.external_services.vector_stores.weaviate_client import (
    WeaviateClient,
)

logger = logging.getLogger(__name__)


class LlamaIndexRetriever(BaseRetriever):
    """Flexible wrapper for any LlamaIndex BaseRetriever.

    This wrapper provides LangChain BaseRetriever compatibility for any
    LlamaIndex retriever, converting NodeWithScore results to LangChain
    Document objects with proper async support and configurable behavior.

    Attributes:
        llama_retriever: The underlying LlamaIndex BaseRetriever instance
        config: Optional configuration dictionary for custom behavior
    """

    llama_retriever: Any  # LlamaIndex BaseRetriever - using Any for flexibility
    config: Dict[str, Any] = Field(default_factory=dict)

    def __init__(
        self,
        llama_retriever: Any,  # Any LlamaIndex retriever instance
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the LlamaIndex retriever wrapper.

        Args:
            llama_retriever: A LlamaIndex BaseRetriever instance
            config: Optional configuration for custom behavior
        """
        super().__init__(
            llama_retriever=llama_retriever,
            config=config or {}
        )

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None,
    ) -> List[Document]:
        """Retrieve documents synchronously.

        Args:
            query: Search query string
            run_manager: Optional callback manager (unused)

        Returns:
            List of LangChain Document objects with content and metadata
        """
        nodes = self.llama_retriever.retrieve(query)
        return self._nodes_to_documents(nodes)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None,
    ) -> List[Document]:
        """Retrieve documents asynchronously.

        Uses sync retrieve() with run_in_executor() because the underlying
        WeaviateClient uses a synchronous Weaviate v4 client, not WeaviateAsyncClient.

        Args:
            query: Search query string
            run_manager: Optional callback manager (unused)

        Returns:
            List of LangChain Document objects with content and metadata
        """
        loop = asyncio.get_event_loop()
        nodes = await loop.run_in_executor(
            None, lambda: self.llama_retriever.retrieve(query)
        )
        return self._nodes_to_documents(nodes)

    
    def _nodes_to_documents(self, nodes: List[Any]) -> List[Document]:
        """Convert LlamaIndex NodeWithScore objects to LangChain Documents.

        Args:
            nodes: List of LlamaIndex NodeWithScore objects

        Returns:
            List of LangChain Document objects
        """
        documents = []
        for node in nodes:
            # Extract content from the node
            content = node.node.get_content()

            # Build metadata including similarity score
            metadata = dict(node.node.metadata) if node.node.metadata else {}
            metadata["_distance"] = node.score if node.score is not None else 0.0

            documents.append(
                Document(
                    page_content=content,
                    metadata=metadata,
                )
            )

        return documents


def create_llama_index_weaviate_retriever(
    weaviate_client: WeaviateClient,
    top_k: int = 5,
    similarity_threshold: float = 0.7,
    index_name: str = "Chunk",
    text_key: str = "text",
    embed_model_name: str = "Alibaba-NLP/gte-multilingual-base",
) -> BaseRetriever:
    """Create LlamaIndex-backed retriever with LangChain interface.

    Uses LlamaIndex's VectorStoreIndex for native Weaviate integration,
    wrapped in a custom adapter for LangChain BaseRetriever compatibility.

    Args:
        weaviate_client: Connected WeaviateClient instance
        top_k: Number of chunks to retrieve (default: 5)
        similarity_threshold: Minimum similarity score 0.0-1.0 (default: 0.7)
        index_name: Weaviate collection name (default: "Chunk")
        text_key: Property name for chunk text (default: "text")
        embed_model_name: HuggingFace embedding model (default: Alibaba-NLP/gte-multilingual-base)

    Returns:
        LangChain-compatible BaseRetriever using LlamaIndex backend

    Raises:
        ValueError: If weaviate_client is not connected

    Examples:
        >>> from app.infrastructure.external_services.vector_stores.weaviate_client import WeaviateClient
        >>> from app.config.vector_stores.weaviate_config import WeaviateConfig
        >>>
        >>> config = WeaviateConfig()
        >>> client = WeaviateClient(config)
        >>> await client.connect()
        >>>
        >>> retriever = create_llama_index_weaviate_retriever(client, top_k=5)
        >>> documents = await retriever.ainvoke("fitness query")
    """
    # Validate connection
    if not weaviate_client.is_connected:
        raise ValueError(
            "WeaviateClient must be connected before creating retriever. "
            "Call await weaviate_client.connect() first."
        )

    logger.info(
        "Creating LlamaIndex retriever: index=%s, model=%s, top_k=%d",
        index_name,
        embed_model_name,
        top_k,
    )

    try:
        # Initialize Weaviate vector store with LlamaIndex
        vector_store = WeaviateVectorStore(
            weaviate_client=weaviate_client._client,  # Access native Weaviate v4 client
            index_name=index_name,
            text_key=text_key,
        )

        # Create embedding model (reuses HuggingFace transformers)
        embed_model = HuggingFaceEmbedding(
            model_name=embed_model_name,
            trust_remote_code=True,  # Required for Alibaba-NLP models
        )

        # Create index from existing vector store (no re-indexing)
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
        )

        # Use LlamaIndex's native as_retriever() method
        llama_retriever: LlamaIndexBaseRetriever = index.as_retriever(similarity_top_k=top_k)

        # Create configuration for the wrapper
        config = {
            "similarity_threshold": similarity_threshold,
            "top_k": top_k,
            "index_name": index_name,
            "text_key": text_key,
        }

        # Wrap in LangChain-compatible retriever
        retriever = LlamaIndexRetriever(llama_retriever=llama_retriever, config=config)

        logger.info("LlamaIndex retriever created successfully")

        return retriever

    except Exception as e:
        logger.error("Failed to create LlamaIndex retriever: %s", str(e))
        raise
