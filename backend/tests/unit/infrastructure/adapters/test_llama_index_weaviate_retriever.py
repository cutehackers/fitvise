"""Unit tests for LlamaIndex Weaviate retriever adapter."""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from app.infrastructure.adapters.llama_index_weaviate_retriever import (
    create_llama_index_retriever,
)
from app.infrastructure.external_services.vector_stores.weaviate_client import (
    WeaviateClient,
)


class TestCreateLlamaIndexRetriever:
    """Test suite for create_llama_index_retriever factory function."""

    def test_raises_on_disconnected_client(self):
        """Test retriever raises ValueError if WeaviateClient is not connected."""
        # Arrange
        mock_client = Mock(spec=WeaviateClient)
        mock_client.is_connected = False

        # Act & Assert
        with pytest.raises(ValueError, match="must be connected"):
            create_llama_index_retriever(mock_client)

    @patch("app.infrastructure.adapters.llama_index_weaviate_retriever.WeaviateVectorStore")
    @patch("app.infrastructure.adapters.llama_index_weaviate_retriever.HuggingFaceEmbedding")
    @patch("app.infrastructure.adapters.llama_index_weaviate_retriever.VectorStoreIndex")
    @patch("app.infrastructure.adapters.llama_index_weaviate_retriever.LlamaIndexRetriever")
    def test_create_retriever_success(
        self,
        mock_llama_retriever,
        mock_index,
        mock_embedding,
        mock_vector_store,
    ):
        """Test successful retriever creation with mocked components."""
        # Arrange
        mock_client = Mock(spec=WeaviateClient)
        mock_client.is_connected = True
        mock_client._client = Mock()  # Native Weaviate v4 client

        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance

        mock_embedding_instance = Mock()
        mock_embedding.return_value = mock_embedding_instance

        mock_index_instance = Mock()
        mock_index.from_vector_store.return_value = mock_index_instance

        mock_retriever_instance = Mock()
        mock_llama_retriever.return_value = mock_retriever_instance

        # Act
        retriever = create_llama_index_retriever(
            weaviate_client=mock_client,
            top_k=5,
            index_name="Chunk",
            text_key="text",
            embed_model_name="Alibaba-NLP/gte-multilingual-base",
        )

        # Assert
        assert retriever == mock_retriever_instance

        # Verify WeaviateVectorStore called correctly
        mock_vector_store.assert_called_once_with(
            weaviate_client=mock_client._client,
            index_name="Chunk",
            text_key="text",
        )

        # Verify HuggingFaceEmbedding called correctly
        mock_embedding.assert_called_once_with(
            model_name="Alibaba-NLP/gte-multilingual-base",
            trust_remote_code=True,
        )

        # Verify VectorStoreIndex created correctly
        mock_index.from_vector_store.assert_called_once_with(
            vector_store=mock_vector_store_instance,
            embed_model=mock_embedding_instance,
        )

        # Verify LlamaIndexRetriever created correctly
        mock_llama_retriever.assert_called_once_with(
            index=mock_index_instance,
            query_kwargs={"similarity_top_k": 5},
        )

    @patch("app.infrastructure.adapters.llama_index_weaviate_retriever.WeaviateVectorStore")
    @patch("app.infrastructure.adapters.llama_index_weaviate_retriever.HuggingFaceEmbedding")
    @patch("app.infrastructure.adapters.llama_index_weaviate_retriever.VectorStoreIndex")
    @patch("app.infrastructure.adapters.llama_index_weaviate_retriever.LlamaIndexRetriever")
    def test_create_retriever_with_custom_params(
        self,
        mock_llama_retriever,
        mock_index,
        mock_embedding,
        mock_vector_store,
    ):
        """Test retriever creation with custom parameters."""
        # Arrange
        mock_client = Mock(spec=WeaviateClient)
        mock_client.is_connected = True
        mock_client._client = Mock()

        mock_vector_store.return_value = Mock()
        mock_embedding.return_value = Mock()
        mock_index.from_vector_store.return_value = Mock()
        mock_llama_retriever.return_value = Mock()

        # Act
        retriever = create_llama_index_retriever(
            weaviate_client=mock_client,
            top_k=10,
            similarity_threshold=0.8,
            index_name="CustomChunk",
            text_key="custom_text",
            embed_model_name="custom-model",
        )

        # Assert
        assert retriever is not None

        # Verify custom parameters used
        mock_vector_store.assert_called_once_with(
            weaviate_client=mock_client._client,
            index_name="CustomChunk",
            text_key="custom_text",
        )

        mock_embedding.assert_called_once_with(
            model_name="custom-model",
            trust_remote_code=True,
        )

        mock_llama_retriever.assert_called_once()
        call_kwargs = mock_llama_retriever.call_args[1]
        assert call_kwargs["query_kwargs"]["similarity_top_k"] == 10

    @patch("app.infrastructure.adapters.llama_index_weaviate_retriever.WeaviateVectorStore")
    def test_create_retriever_handles_vector_store_error(self, mock_vector_store):
        """Test retriever creation handles vector store initialization errors."""
        # Arrange
        mock_client = Mock(spec=WeaviateClient)
        mock_client.is_connected = True
        mock_client._client = Mock()

        mock_vector_store.side_effect = Exception("Vector store init failed")

        # Act & Assert
        with pytest.raises(Exception, match="Vector store init failed"):
            create_llama_index_retriever(mock_client)

    @patch("app.infrastructure.adapters.llama_index_weaviate_retriever.WeaviateVectorStore")
    @patch("app.infrastructure.adapters.llama_index_weaviate_retriever.HuggingFaceEmbedding")
    def test_create_retriever_handles_embedding_error(
        self, mock_embedding, mock_vector_store
    ):
        """Test retriever creation handles embedding model initialization errors."""
        # Arrange
        mock_client = Mock(spec=WeaviateClient)
        mock_client.is_connected = True
        mock_client._client = Mock()

        mock_vector_store.return_value = Mock()
        mock_embedding.side_effect = Exception("Embedding model load failed")

        # Act & Assert
        with pytest.raises(Exception, match="Embedding model load failed"):
            create_llama_index_retriever(mock_client)
