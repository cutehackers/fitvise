from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from llama_index.core.vector_stores.types import FilterOperator, VectorStoreQueryMode

from botadvisor.app.core.entity.chunk import Chunk
from botadvisor.app.core.entity.document_metadata import DocumentMetadata
from botadvisor.app.core.entity.retriever_request import RetrieverRequest


def test_retrieval_config_defaults_to_hybrid_mode():
    from botadvisor.app.retrieval.config import RetrievalConfig

    config = RetrievalConfig()

    assert config.index_name == "BotAdvisorDocs"
    assert config.query_mode is VectorStoreQueryMode.HYBRID
    assert config.alpha == 0.5
    assert config.similarity_top_k == 5


def test_factory_rejects_disconnected_weaviate_client():
    from botadvisor.app.retrieval.config import RetrievalConfig
    from botadvisor.app.retrieval.factory import create_hybrid_retriever

    mock_client = Mock()
    mock_client.is_connected = False

    with pytest.raises(ValueError, match="must be connected"):
        create_hybrid_retriever(mock_client, RetrievalConfig())


@patch("botadvisor.app.retrieval.factory.VectorStoreIndex")
@patch("botadvisor.app.retrieval.factory.HuggingFaceEmbedding")
@patch("botadvisor.app.retrieval.factory.WeaviateVectorStore")
@patch("botadvisor.app.retrieval.service.VectorIndexRetriever")
def test_retriever_applies_hybrid_mode_and_platform_filter(
    mock_vector_index_retriever,
    mock_vector_store,
    mock_embedding,
    mock_vector_store_index,
):
    from botadvisor.app.retrieval.config import RetrievalConfig
    from botadvisor.app.retrieval.factory import create_hybrid_retriever

    mock_client = Mock()
    mock_client.is_connected = True
    mock_client._client = Mock()

    mock_vector_store_instance = Mock()
    mock_vector_store.return_value = mock_vector_store_instance

    mock_embedding.return_value = Mock()

    mock_index = Mock()
    mock_vector_store_index.from_vector_store.return_value = mock_index

    mock_node = Mock()
    mock_node.node.get_content.return_value = "retrieved content"
    mock_node.node.metadata = {
        "doc_id": "doc-1",
        "source_id": "file-1",
        "platform": "filesystem",
        "source_url": "file:///tmp/sample.txt",
        "section": "chunk_0",
    }
    mock_node.score = 0.91

    mock_li_retriever = Mock()
    mock_li_retriever.retrieve.return_value = [mock_node]
    mock_vector_index_retriever.return_value = mock_li_retriever

    service = create_hybrid_retriever(mock_client, RetrievalConfig())
    results = service.retrieve(RetrieverRequest(query="protein intake", platform="filesystem", top_k=3))

    assert len(results) == 1
    assert results[0].content == "retrieved content"
    assert results[0].metadata == DocumentMetadata(
        doc_id="doc-1",
        source_id="file-1",
        platform="filesystem",
        source_url="file:///tmp/sample.txt",
        page=None,
        section="chunk_0",
    )
    assert results[0].score == 0.91

    mock_vector_index_retriever.assert_called_once()
    kwargs = mock_vector_index_retriever.call_args.kwargs
    assert kwargs["index"] is mock_index
    assert kwargs["similarity_top_k"] == 3
    assert kwargs["vector_store_query_mode"] is VectorStoreQueryMode.HYBRID
    assert kwargs["alpha"] == 0.5
    assert kwargs["sparse_top_k"] == 3
    assert kwargs["hybrid_top_k"] == 3

    filters = kwargs["filters"]
    assert filters is not None
    assert len(filters.filters) == 1
    assert filters.filters[0].key == "platform"
    assert filters.filters[0].value == "filesystem"
    assert filters.filters[0].operator is FilterOperator.EQ


def test_langchain_adapter_preserves_citation_metadata():
    from botadvisor.app.retrieval.langchain_adapter import LangChainRetrieverAdapter

    retrieval_service = Mock()
    retrieval_service.retrieve.return_value = [
        Chunk(
            chunk_id="doc-1_chunk_0",
            content="retrieved content",
            metadata=DocumentMetadata(
                doc_id="doc-1",
                source_id="file-1",
                platform="filesystem",
                source_url="file:///tmp/sample.txt",
                page=2,
                section="chunk_0",
            ),
            score=0.73,
        )
    ]

    adapter = LangChainRetrieverAdapter(retrieval_service=retrieval_service)

    documents = adapter.invoke("protein intake")

    assert len(documents) == 1
    assert documents[0].page_content == "retrieved content"
    assert documents[0].metadata == {
        "chunk_id": "doc-1_chunk_0",
        "doc_id": "doc-1",
        "source_id": "file-1",
        "platform": "filesystem",
        "source_url": "file:///tmp/sample.txt",
        "page": 2,
        "section": "chunk_0",
        "_distance": 0.73,
    }
    retrieval_service.retrieve.assert_called_once_with(RetrieverRequest(query="protein intake"))
