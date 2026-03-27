"""Unit tests for LlamaIndex Weaviate retriever."""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from app.infrastructure.llm.services.llama_index_retriever import (
    create_llama_index_weaviate_retriever,
    LlamaIndexRetriever,
)
from app.infrastructure.external_services.vector_stores.weaviate_client import (
    WeaviateClient,
)


class TestCreateLlamaIndexRetriever:
    """Test suite for create_llama_index_weaviate_retriever factory function."""

    def test_raises_on_disconnected_client(self):
        """Test retriever raises ValueError if WeaviateClient is not connected."""
        # Arrange
        mock_client = Mock(spec=WeaviateClient)
        mock_client.is_connected = False

        # Act & Assert
        with pytest.raises(ValueError, match="must be connected"):
            create_llama_index_weaviate_retriever(mock_client)

    @patch("app.infrastructure.llm.services.llama_index_retriever.WeaviateVectorStore")
    @patch("app.infrastructure.llm.services.llama_index_retriever.HuggingFaceEmbedding")
    @patch("app.infrastructure.llm.services.llama_index_retriever.VectorStoreIndex")
    @patch("app.infrastructure.llm.services.llama_index_retriever.LlamaIndexRetriever")
    def test_create_retriever_success(
        self,
        mock_wrapper_retriever,
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

        mock_llama_retriever_instance = Mock()
        mock_index_instance.as_retriever.return_value = mock_llama_retriever_instance

        mock_wrapper_retriever_instance = Mock()
        mock_wrapper_retriever.return_value = mock_wrapper_retriever_instance

        # Act
        retriever = create_llama_index_weaviate_retriever(
            weaviate_client=mock_client,
            top_k=5,
            index_name="Chunk",
            text_key="text",
            embed_model_name="Alibaba-NLP/gte-multilingual-base",
        )

        # Assert
        assert retriever == mock_wrapper_retriever_instance

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

        # Verify LlamaIndex retriever created correctly
        mock_index_instance.as_retriever.assert_called_once_with(similarity_top_k=5)

        # Verify wrapper retriever created correctly
        mock_wrapper_retriever.assert_called_once_with(
            llama_retriever=mock_llama_retriever_instance,
            config={
                "similarity_threshold": 0.7,
                "top_k": 5,
                "index_name": "Chunk",
                "text_key": "text",
            },
        )

    @patch("app.infrastructure.llm.services.llama_index_retriever.WeaviateVectorStore")
    @patch("app.infrastructure.llm.services.llama_index_retriever.HuggingFaceEmbedding")
    @patch("app.infrastructure.llm.services.llama_index_retriever.VectorStoreIndex")
    @patch("app.infrastructure.llm.services.llama_index_retriever.LlamaIndexRetriever")
    def test_create_retriever_with_custom_params(
        self,
        mock_wrapper_retriever,
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

        mock_index_instance = Mock()
        mock_index.from_vector_store.return_value = mock_index_instance

        mock_llama_retriever_instance = Mock()
        mock_index_instance.as_retriever.return_value = mock_llama_retriever_instance

        mock_wrapper_retriever.return_value = Mock()

        # Act
        retriever = create_llama_index_weaviate_retriever(
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

        # Verify LlamaIndex retriever created with custom top_k
        mock_index_instance.as_retriever.assert_called_once_with(similarity_top_k=10)

        # Verify wrapper retriever created with custom config
        mock_wrapper_retriever.assert_called_once()
        call_args = mock_wrapper_retriever.call_args
        assert call_args[1]["config"]["top_k"] == 10
        assert call_args[1]["config"]["similarity_threshold"] == 0.8
        assert call_args[1]["config"]["index_name"] == "CustomChunk"
        assert call_args[1]["config"]["text_key"] == "custom_text"

    @patch("app.infrastructure.llm.services.llama_index_retriever.WeaviateVectorStore")
    def test_create_retriever_handles_vector_store_error(self, mock_vector_store):
        """Test retriever creation handles vector store initialization errors."""
        # Arrange
        mock_client = Mock(spec=WeaviateClient)
        mock_client.is_connected = True
        mock_client._client = Mock()

        mock_vector_store.side_effect = Exception("Vector store init failed")

        # Act & Assert
        with pytest.raises(Exception, match="Vector store init failed"):
            create_llama_index_weaviate_retriever(mock_client)

    @patch("app.infrastructure.llm.services.llama_index_retriever.WeaviateVectorStore")
    @patch("app.infrastructure.llm.services.llama_index_retriever.HuggingFaceEmbedding")
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
            create_llama_index_weaviate_retriever(mock_client)


class TestLlamaIndexRetriever:
    """Test suite for LlamaIndexRetriever wrapper class."""

    def test_init_with_default_config(self):
        """Test retriever initialization with default configuration."""
        # Arrange
        mock_llama_retriever = Mock()

        # Act
        retriever = LlamaIndexRetriever(mock_llama_retriever)

        # Assert
        assert retriever.llama_retriever == mock_llama_retriever
        assert retriever.config == {}

    def test_init_with_custom_config(self):
        """Test retriever initialization with custom configuration."""
        # Arrange
        mock_llama_retriever = Mock()
        custom_config = {"debug": True, "max_results": 3}

        # Act
        retriever = LlamaIndexRetriever(mock_llama_retriever, custom_config)

        # Assert
        assert retriever.llama_retriever == mock_llama_retriever
        assert retriever.config == custom_config

    @pytest.mark.asyncio
    async def test_base_retriever_ainvoke_with_string_input(self):
        """Test BaseRetriever's default ainvoke method with string input."""
        # Arrange
        mock_llama_retriever = Mock()
        mock_node = Mock()
        mock_node.node.get_content.return_value = "Test content"
        mock_node.node.metadata = {"source": "test"}
        mock_node.score = 0.8
        mock_llama_retriever.retrieve.return_value = [mock_node]

        retriever = LlamaIndexRetriever(mock_llama_retriever)

        # Act - Using BaseRetriever's default ainvoke implementation
        result = await retriever.ainvoke("test query")

        # Assert
        assert len(result) == 1
        assert result[0].page_content == "Test content"
        assert result[0].metadata["source"] == "test"
        assert result[0].metadata["_distance"] == 0.8
        mock_llama_retriever.retrieve.assert_called_once_with("test query")

    @pytest.mark.asyncio
    async def test_base_retriever_ainvoke_with_invalid_input_type(self):
        """Test BaseRetriever raises error for non-string input types."""
        # Arrange
        mock_llama_retriever = Mock()
        retriever = LlamaIndexRetriever(mock_llama_retriever)

        # Act & Assert - BaseRetriever will fail with non-string inputs
        # since _aget_relevant_documents expects a string query
        with pytest.raises((TypeError, AttributeError)):
            await retriever.ainvoke(123)

    def test_get_relevant_documents_sync(self):
        """Test synchronous document retrieval."""
        # Arrange
        mock_llama_retriever = Mock()
        mock_node = Mock()
        mock_node.node.get_content.return_value = "Test content"
        mock_node.node.metadata = {"source": "test"}
        mock_node.score = 0.8
        mock_llama_retriever.retrieve.return_value = [mock_node]

        retriever = LlamaIndexRetriever(mock_llama_retriever)

        # Act
        result = retriever._get_relevant_documents("test query")

        # Assert
        assert len(result) == 1
        assert result[0].page_content == "Test content"
        assert result[0].metadata["source"] == "test"
        assert result[0].metadata["_distance"] == 0.8

    def test_nodes_to_documents_with_empty_metadata(self):
        """Test _nodes_to_documents with nodes without metadata."""
        # Arrange
        mock_nodes = [Mock()]
        mock_nodes[0].node.get_content.return_value = "Test content"
        mock_nodes[0].node.metadata = None
        mock_nodes[0].score = 0.5

        retriever = LlamaIndexRetriever(Mock())

        # Act
        result = retriever._nodes_to_documents(mock_nodes)

        # Assert
        assert len(result) == 1
        assert result[0].page_content == "Test content"
        assert result[0].metadata == {"_distance": 0.5}
