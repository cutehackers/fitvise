"""Test LangFuse integration with callback handlers."""

import pytest
import os
from unittest.mock import AsyncMock, MagicMock, patch
from app.core.settings import Settings
from app.infrastructure.analytics.langchain_callbacks import LangFuseCallbackHandler
from app.infrastructure.external_services.ml_services.llm_services.ollama_service import OllamaService
from app.infrastructure.llm.services.llama_index_retriever import LlamaIndexRetriever


class TestLangFuseIntegration:
    """Test suite for LangFuse integration with usage analytics."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        return Settings(
            # App Information
            app_name="test-app",
            app_version="1.0.0",
            app_description="Test app",

            # Environment
            environment="local",
            debug=True,
            domain="localhost",
            api_host="localhost",
            api_port=8000,

            # LLM
            llm_base_url="http://localhost:11434",
            llm_model="llama2",
            llm_timeout=30,
            llm_temperature=0.7,
            llm_max_tokens=1000,

            # LangFuse - use environment variables for testing
            langfuse_secret_key="test-secret-key-1234567890",
            langfuse_public_key="test-public-key-1234567890",
            langfuse_host="http://localhost:3000",

            # Minimal required fields
            database_url="sqlite:///test.db",
            vector_store_type="weaviate",
            embedding_model="test-model",
            vector_dimension=384,
            secret_key="test-secret-key-12345678901234567890",
            access_token_expire_minutes=30,
            algorithm="HS256",
            max_file_size=10485760,
            allowed_file_types="pdf,txt,docx",
            upload_directory="uploads",
            knowledge_base_path="knowledge_base",
            auto_index_on_startup=False,
            index_update_interval=3600,
            log_level="INFO",
            log_file="app.log",
            log_rotation="daily",
            log_retention="30 days",
            rag_retrieval_top_k=5,
            rag_retrieval_similarity_threshold=0.7,
            llm_context_window=4096,
            llm_reserve_tokens=512,
            context_truncation_strategy="recent",
            llm_max_concurrent=10,
            health_check_interval=60,
            health_min_success_rate=95.0,
            health_max_response_time_ms=5000.0,

            # CORS defaults
            cors_origins="*",
            cors_allow_credentials=True,
            cors_allow_methods="GET,POST,PUT,DELETE,OPTIONS",
            cors_allow_headers="*",

            # Weaviate defaults
            weaviate_host="localhost",
            weaviate_port=8080,
            weaviate_grpc_port=50051,
            weaviate_timeout=30,
            weaviate_batch_size=100,

            # Sentence transformer defaults
            sentence_transformer_model_name="all-MiniLM-L6-v2",
            sentence_transformer_device="cpu",
            sentence_transformer_cache_folder="./cache",
            sentence_transformer_dimension=384,
            sentence_transformer_batch_size=32,
            sentence_transformer_cache_size_mb=1024,

            # Framework tracing defaults
            llamaindex_tracing_enabled=False,  # Disabled for simple approach
            langchain_tracing_enabled=False,   # Disabled for simple approach
            framework_tracing_sample_rate=1.0,
            framework_tracing_privacy_masking=True,
            framework_tracing_max_payload_size=1024,
            llamaindex_trace_document_loading=True,
            llamaindex_trace_vector_operations=True,
            llamaindex_trace_embedding_generation=True,
            llamaindex_trace_retrieval_operations=True,
            llamaindex_max_chunk_size_for_tracing=1000,
        )

    @pytest.mark.asyncio
    async def test_callback_handler_initialization(self, mock_settings):
        """Test that callback handler initializes correctly with settings."""
        with patch('app.infrastructure.analytics.langchain_callbacks.Langfuse') as mock_langfuse, \
             patch('app.infrastructure.analytics.langchain_callbacks.LangfuseLangchainHandler') as mock_lc_handler:
            mock_client = MagicMock()
            mock_langfuse.return_value = mock_client
            mock_lc_handler.return_value = MagicMock()

            handler = LangFuseCallbackHandler(
                secret_key=mock_settings.langfuse_secret_key,
                public_key=mock_settings.langfuse_public_key,
                host=mock_settings.langfuse_host
            )

            mock_langfuse.assert_called_once_with(
                secret_key=mock_settings.langfuse_secret_key,
                public_key=mock_settings.langfuse_public_key,
                host=mock_settings.langfuse_host
            )
            mock_lc_handler.assert_called_once()
            assert handler.is_enabled() is True

    def test_callback_handler_disabled(self):
        """Test that callback handler handles disabled configuration."""
        with patch('app.infrastructure.analytics.langchain_callbacks.Langfuse', side_effect=Exception("missing keys")):
            handler = LangFuseCallbackHandler()
            assert handler.is_enabled() is False

    def test_callback_handler_enabled(self):
        """Test that callback handler works when environment variables are set."""
        with patch.dict(os.environ, {
            'LANGFUSE_SECRET_KEY': 'test-secret-key',
            'LANGFUSE_PUBLIC_KEY': 'test-public-key',
            'LANGFUSE_HOST': 'http://localhost:3000'
        }), patch('app.infrastructure.analytics.langchain_callbacks.Langfuse') as mock_langfuse, \
             patch('app.infrastructure.analytics.langchain_callbacks.LangfuseLangchainHandler') as mock_lc_handler:
            mock_langfuse.return_value = MagicMock()
            mock_lc_handler.return_value = MagicMock()

            handler = LangFuseCallbackHandler()

            assert handler.is_enabled() is True

    @pytest.mark.asyncio
    async def test_ollama_service_with_callback_handler(self, mock_settings):
        """Test OllamaService with callback handler injection."""
        with patch('app.infrastructure.analytics.langchain_callbacks.Langfuse') as mock_langfuse, \
             patch('app.infrastructure.analytics.langchain_callbacks.LangfuseLangchainHandler') as mock_lc_handler:
            mock_langfuse.return_value = MagicMock()
            mock_lc_handler.return_value = MagicMock()

            callback_handler = LangFuseCallbackHandler(
                secret_key=mock_settings.langfuse_secret_key,
                public_key=mock_settings.langfuse_public_key,
                host=mock_settings.langfuse_host
            )

            ollama_service = OllamaService(
                settings=mock_settings,
                callback_handler=callback_handler
            )

            assert ollama_service._llm.callbacks is not None
            assert callback_handler in ollama_service._llm.callbacks

    @pytest.mark.asyncio
    async def test_ollama_service_without_callback_handler(self, mock_settings):
        """Test OllamaService without callback handler injection."""
        ollama_service = OllamaService(settings=mock_settings)

        # Verify no callbacks are set
        assert ollama_service._llm.callbacks is None

    @pytest.mark.asyncio
    async def test_retriever_with_callback_handler(self, mock_settings):
        """Test LlamaIndexRetriever with callback handler injection."""
        with patch('app.infrastructure.analytics.langchain_callbacks.Langfuse') as mock_langfuse, \
             patch('app.infrastructure.analytics.langchain_callbacks.LangfuseLangchainHandler') as mock_lc_handler:
            mock_langfuse.return_value = MagicMock()
            mock_lc_handler.return_value = MagicMock()

            callback_handler = LangFuseCallbackHandler(
                secret_key=mock_settings.langfuse_secret_key,
                public_key=mock_settings.langfuse_public_key,
                host=mock_settings.langfuse_host
            )

            mock_llama_retriever = MagicMock()
            retriever = LlamaIndexRetriever(llama_retriever=mock_llama_retriever)

            # Ensure retriever still works with callback handler available (callbacks are passed at invocation time)
            assert retriever is not None
            assert callback_handler.is_enabled()

    @pytest.mark.asyncio
    async def test_retriever_without_callback_handler(self, mock_settings):
        """Test LlamaIndexRetriever without callback handler injection."""
        mock_llama_retriever = MagicMock()

        retriever = LlamaIndexRetriever(llama_retriever=mock_llama_retriever)

        assert retriever is not None

    def test_settings_langfuse_configuration(self, mock_settings):
        """Test LangFuse configuration validation."""
        # Test valid configuration
        assert mock_settings.langfuse_configured is True
        assert mock_settings.langfuse_secret_key == "test-secret-key-1234567890"
        assert mock_settings.langfuse_public_key == "test-public-key-1234567890"
        assert mock_settings.langfuse_host == "http://localhost:3000"

    def test_settings_langfuse_not_configured(self):
        """Test LangFuse when not configured."""
        settings = Settings.model_construct(
            langfuse_secret_key="",
            langfuse_public_key="",
            langfuse_host="https://cloud.langfuse.com"
        )

        assert settings.langfuse_configured is False

    def test_langfuse_configured_property(self):
        """Test langfuse_configured property logic."""
        # Both keys present
        settings = Settings.model_construct(
            langfuse_secret_key="sk-test",
            langfuse_public_key="pk-test"
        )
        assert settings.langfuse_configured is True

        # Missing secret key
        settings = Settings.model_construct(
            langfuse_secret_key="",
            langfuse_public_key="pk-test"
        )
        assert settings.langfuse_configured is False

        # Missing public key
        settings = Settings.model_construct(
            langfuse_secret_key="sk-test",
            langfuse_public_key=""
        )
        assert settings.langfuse_configured is False

        # Both keys missing
        settings = Settings.model_construct(
            langfuse_secret_key="",
            langfuse_public_key=""
        )
        assert settings.langfuse_configured is False
