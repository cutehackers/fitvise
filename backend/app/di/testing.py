"""Testing Support for DI System.

This module provides testing utilities and mock providers for the DI system.
It enables easy dependency overriding and test isolation.
"""

from typing import Optional, Any
from unittest.mock import AsyncMock, MagicMock

from dependency_injector import containers, providers

from app.di.container import FitviseContainer
from app.di.providers.config import ConfigProviders
from app.di.providers.external import ExternalServiceProviders
from app.di.providers.repositories import RepositoryProviders
from app.di.providers.services import ServiceProviders


class MockConfigProviders(containers.DeclarativeContainer):
    """Mock configuration providers for testing."""
    
    # Mock settings
    settings = providers.Singleton(
        lambda: MagicMock(
            app_name="FitVise Test",
            app_version="1.0.0-test",
            app_description="Test environment",
            environment="test",
            debug=True,
            api_host="127.0.0.1",
            api_port=8001,
            llm_base_url="http://localhost:11434",
            llm_model="test-model",
            llm_timeout=30,
            llm_temperature=0.1,
            llm_max_tokens=1000,
            api_v1_prefix="/api/v1",
            cors_origins="*",
            cors_allow_credentials=True,
            cors_allow_methods="*",
            cors_allow_headers="*",
            database_url="sqlite:///:memory:",
            database_echo=False,
            vector_store_type="chromadb",
            vector_store_path="/tmp/test_vectors",
            embedding_model="test-embedding",
            vector_dimension=384,
            secret_key="test-secret-key",
            access_token_expire_minutes=60,
            algorithm="HS256",
            max_file_size=1048576,
            allowed_file_types="txt,pdf,docx",
            upload_directory="/tmp/test_uploads",
            knowledge_base_path="/tmp/test_knowledge",
            auto_index_on_startup=False,
            index_update_interval=3600,
            log_level="DEBUG",
            log_file="/tmp/test.log",
            log_rotation="daily",
            log_retention="7 days",
            # Add any other required settings
        )
    )

    # Mock configuration objects
    embedding_config = providers.Singleton(
        lambda: MagicMock(
            model_name="test-embedding",
            device="cpu",
            batch_size=1,
            show_progress=False,
        )
    )

    weaviate_config = providers.Singleton(
        lambda: MagicMock(
            host="localhost",
            port=8080,
            scheme="http",
        )
    )

    # Test-specific flags
    is_production = providers.Factory(lambda: False)
    is_development = providers.Factory(lambda: True)
    debug_enabled = providers.Factory(lambda: True)


class MockExternalServiceProviders(containers.DeclarativeContainer):
    """Mock external service providers for testing."""
    
    config = providers.Dependency()
    
    # Mock Weaviate client
    weaviate_client = providers.Singleton(
        lambda: MagicMock(
            is_connected=True,
            health_check=AsyncMock(return_value=True),
        )
    )

    # Mock sentence transformer service
    sentence_transformer_service = providers.Singleton(
        lambda: MagicMock(
            is_loaded=True,
            initialize=AsyncMock(),
            embed_text=AsyncMock(return_value=[0.1, 0.2, 0.3]),
            embed_batch=AsyncMock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            health_check=AsyncMock(return_value=True),
        )
    )

    # Mock Ollama service
    ollama_service = providers.Singleton(
        lambda: MagicMock(
            is_connected=True,
            initialize=AsyncMock(),
            generate_response=AsyncMock(return_value="Test response"),
            health_check=AsyncMock(return_value=True),
        )
    )

    # Mock embedding model
    llama_index_embedding = providers.Singleton(
        lambda: MagicMock(
            get_text_embedding=lambda text: [0.1, 0.2, 0.3],
            get_text_embeddings=lambda texts: [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        )
    )

    # Mock health checks
    weaviate_health_check = providers.Factory(
        lambda: AsyncMock(return_value=True)
    )
    
    embedding_service_health_check = providers.Factory(
        lambda: AsyncMock(return_value=True)
    )
    
    ollama_health_check = providers.Factory(
        lambda: AsyncMock(return_value=True)
    )
    
    external_services_health = providers.Factory(
        lambda weaviate=True, embedding=True, ollama=True: {
            "overall": weaviate and embedding and ollama,
            "weaviate": weaviate,
            "embedding_service": embedding,
            "ollama": ollama,
        }
    )


class MockRepositoryProviders(containers.DeclarativeContainer):
    """Mock repository providers for testing."""
    
    config = providers.Dependency()
    external_services = providers.Dependency()
    
    # Mock async session
    session = providers.Singleton(
        lambda: AsyncMock()
    )

    async def _transaction_session():
        mock_session = AsyncMock()
        yield mock_session

    transaction_session = providers.Resource(_transaction_session)

    # Mock repositories
    document_repository = providers.Singleton(
        lambda: AsyncMock(
            save=AsyncMock(return_value=MagicMock(id="test-doc-id")),
            find_by_id=AsyncMock(return_value=MagicMock(id="test-doc-id", title="Test Doc")),
            find_all=AsyncMock(return_value=[]),
            delete=AsyncMock(return_value=True),
        )
    )

    data_source_repository = providers.Singleton(
        lambda: AsyncMock(
            save=AsyncMock(return_value=MagicMock(id="test-source-id")),
            find_by_id=AsyncMock(return_value=MagicMock(id="test-source-id", name="Test Source")),
            find_all=AsyncMock(return_value=[]),
        )
    )

    embedding_repository = providers.Singleton(
        lambda: AsyncMock(
            save=AsyncMock(return_value=MagicMock(id="test-embedding-id")),
            find_by_id=AsyncMock(return_value=MagicMock(id="test-embedding-id", vector=[0.1, 0.2, 0.3])),
            search=AsyncMock(return_value=[]),
            delete=AsyncMock(return_value=True),
        )
    )

    # Mock repository container
    repository_container = providers.Singleton(
        lambda document_repo, data_source_repo, embedding_repo: MagicMock(
            document_repository=document_repo,
            data_source_repository=data_source_repo,
            embedding_repository=embedding_repo,
        ),
        document_repo=document_repository,
        data_source_repo=data_source_repository,
        embedding_repo=embedding_repository,
    )

    # Mock repository interfaces
    document_repo_interface = providers.Factory(document_repository)
    data_source_repo_interface = providers.Factory(data_source_repository)
    embedding_repo_interface = providers.Factory(embedding_repository)
    search_repository = providers.Singleton(
        lambda: AsyncMock(
            semantic_search=AsyncMock(return_value=[]),
            find_similar_chunks=AsyncMock(return_value=[]),
            search_by_document_ids=AsyncMock(return_value=[]),
            get_search_suggestions=AsyncMock(return_value=[]),
            aggregate_search_results=AsyncMock(return_value=[]),
            get_popular_queries=AsyncMock(return_value=[]),
            log_search_interaction=AsyncMock(return_value=None),
            health_check=AsyncMock(return_value={"status": "healthy", "weaviate_connected": True, "search_functionality": True}),
            get_search_statistics=AsyncMock(return_value={}),
        )
    )
    search_repo_interface = providers.Factory(search_repository)

    # Mock health checks
    database_health_check = providers.Factory(lambda: AsyncMock(return_value=True))

    async def _repositories_health(database=True, weaviate=True):
        return {
            "overall": database and weaviate,
            "database": database,
            "weaviate": weaviate,
        }

    repositories_health = providers.Factory(_repositories_health)
    repositories_health_check = providers.Factory(_repositories_health)


class MockServiceProviders(containers.DeclarativeContainer):
    """Mock service providers for testing."""
    
    config = providers.Dependency()
    repositories = providers.Dependency()
    external_services = providers.Dependency()
    
    # Mock services
    llm_service = providers.Singleton(
        lambda: MagicMock(
            initialize=AsyncMock(),
            generate_response=AsyncMock(return_value="Test LLM response"),
            health_check=AsyncMock(return_value=True),
        )
    )

    embedding_domain_service = providers.Singleton(
        lambda: AsyncMock(
            create_embedding=AsyncMock(return_value=MagicMock(id="test-embedding-id")),
            find_similar=AsyncMock(return_value=[]),
        )
    )

    # Mock use cases
    prompt_use_case = providers.Singleton(
        lambda llm_service: MagicMock(
            execute=AsyncMock(return_value=MagicMock(
                success=True,
                response="Test response",
                tokens_used=10,
            ))
        ),
        llm_service=llm_service,
    )

    embed_chunks_use_case = providers.Singleton(
        lambda embedding_service, embedding_repo, domain_service: MagicMock(
            execute=AsyncMock(return_value=MagicMock(
                success=True,
                total_chunks=1,
                embedded_count=1,
                stored_count=1,
            ))
        ),
        embedding_service=external_services.sentence_transformer_service,
        embedding_repo=repositories.embedding_repo_interface,
        domain_service=embedding_domain_service,
    )

    retrieval_service = providers.Singleton(
        lambda search_repo, embedding_service: MagicMock(
            semantic_search_with_metrics=AsyncMock(return_value={"results": [], "metrics": {"total_processing_time_ms": 0.0}}),
            find_similar_chunks=AsyncMock(return_value=[]),
            get_search_suggestions=AsyncMock(return_value=[]),
            get_retrieval_metrics=AsyncMock(return_value={}),
        ),
        search_repo=repositories.search_repo_interface,
        embedding_service=external_services.sentence_transformer_service,
    )

    semantic_search_use_case = providers.Singleton(
        lambda retrieval_service: MagicMock(
            execute=AsyncMock(return_value=MagicMock(
                success=True,
                query_id="test-query-id",
                results=[],
                total_results=0,
                processing_time_ms=0.0,
                embedding_time_ms=0.0,
                search_time_ms=0.0,
                query_vector_dimension=0,
                avg_similarity_score=0.0,
                error=None,
                metadata={},
            )),
            find_similar_chunks=AsyncMock(return_value=[]),
            get_search_suggestions=AsyncMock(return_value=[]),
            get_performance_metrics=AsyncMock(return_value={}),
            log_search_feedback=AsyncMock(return_value=None),
        ),
        retrieval_service=retrieval_service,
    )

    setup_airflow_use_case = providers.Singleton(
        lambda: MagicMock(execute=AsyncMock())
    )

    integrate_tika_use_case = providers.Singleton(
        lambda: MagicMock(execute=AsyncMock())
    )

    connect_databases_use_case = providers.Singleton(
        lambda: MagicMock(execute=AsyncMock())
    )

    setup_web_scraping_use_case = providers.Singleton(
        lambda: MagicMock(execute=AsyncMock())
    )

    categorizer = providers.Singleton(
        lambda: MagicMock(
            is_trained=True,
            generate_synthetic_training_data=AsyncMock(return_value=[]),
            train_model=AsyncMock(return_value={"meets_target_accuracy": True}),
            save_model=AsyncMock(),
            load_model=AsyncMock(),
            batch_categorize=AsyncMock(return_value=[]),
            get_model_info=lambda: {"model": "mock"},
        )
    )

    audit_data_sources_use_case = providers.Singleton(
        lambda repo: MagicMock(
            execute=AsyncMock(return_value=MagicMock(
                success=True,
                total_discovered=0,
                total_created=0,
                statistics={},
                export_files=[],
                error_message=None,
            )),
            get_audit_summary=AsyncMock(return_value={}),
        ),
        repo=repositories.data_source_repo_interface,
    )

    document_external_apis_use_case = providers.Singleton(
        lambda repo: MagicMock(
            execute=AsyncMock(return_value=MagicMock(
                success=True,
                total_documented=0,
                total_validated=0,
                created_data_sources=[],
                validation_results={},
                export_files=[],
                error_message=None,
            )),
            get_api_registry_summary=AsyncMock(return_value={}),
        ),
        repo=repositories.data_source_repo_interface,
    )

    categorize_sources_use_case = providers.Singleton(
        lambda repo, categorizer: MagicMock(
            execute=AsyncMock(return_value=MagicMock(
                success=True,
                training_results={"meets_target_accuracy": True},
                total_categorized=0,
                model_info={},
                meets_accuracy_target=True,
                error_message=None,
            )),
            get_categorization_statistics=AsyncMock(return_value={}),
        ),
        repo=repositories.data_source_repo_interface,
        categorizer=categorizer,
    )

    setup_object_storage_use_case = providers.Singleton(
        lambda: MagicMock(
            execute=AsyncMock(
                return_value=MagicMock(
                    provider="local",
                    created_buckets=["test"],
                    environment={},
                    test_put_key="test-key",
                )
            )
        )
    )

    setup_embedding_infrastructure_use_case = providers.Singleton(
        lambda: MagicMock(
            execute=AsyncMock(
                return_value=MagicMock(
                    success=True,
                    embedding_service_status={"is_loaded": True},
                    weaviate_status={"connected": True},
                    schema_created=True,
                    errors=[],
                    as_dict=lambda: {
                        "success": True,
                        "embedding_service": {"is_loaded": True},
                        "weaviate": {"connected": True},
                        "schema_created": True,
                        "errors": [],
                    },
                )
            )
        )
    )

    def _pipeline_repositories(
        document_repo,
        data_source_repo,
        embedding_repo,
    ):
        from app.pipeline.workflow import RepositoryBundle

        return RepositoryBundle(
            document_repository=document_repo,
            data_source_repository=data_source_repo,
            embedding_repository=embedding_repo,
        )

    pipeline_repositories = providers.Factory(
        _pipeline_repositories,
        document_repo=repositories.document_repository,
        data_source_repo=repositories.data_source_repository,
        embedding_repo=repositories.embedding_repository,
    )

    def _pipeline_services(
        embedding_service,
        embedding_model,
        weaviate_client,
        setup_use_case,
    ):
        from app.pipeline.workflow import PipelineServices

        return PipelineServices(
            embedding_service=embedding_service,
            embedding_model=embedding_model,
            weaviate_client=weaviate_client,
            setup_use_case=setup_use_case,
        )

    pipeline_services = providers.Factory(
        _pipeline_services,
        embedding_service=external_services.sentence_transformer_service,
        embedding_model=external_services.llama_index_embedding,
        weaviate_client=external_services.weaviate_client,
        setup_use_case=setup_embedding_infrastructure_use_case,
    )

    # Mock workflow (real instance wired with mocked dependencies)
    def _build_pipeline_workflow(repositories_bundle, services_bundle):
        from app.pipeline.workflow import RAGWorkflow

        return RAGWorkflow(
            repositories=repositories_bundle,
            services=services_bundle,
            verbose=True,
        )

    pipeline_workflow = providers.Factory(
        _build_pipeline_workflow,
        repositories_bundle=pipeline_repositories,
        services_bundle=pipeline_services,
    )

    # Mock health checks
    llm_service_health_check = providers.Factory(lambda: AsyncMock(return_value=True))
    embedding_service_health_check = providers.Factory(lambda: AsyncMock(return_value=True))
    services_health = providers.Factory(
        lambda llm=True, embedding=True: {
            "overall": llm and embedding,
            "llm_service": llm,
            "embedding_service": embedding,
        }
    )


class TestFitviseContainer(containers.DeclarativeContainer):
    """Test container with all mock providers."""
    
    # Mock provider groups
    config = providers.Container(MockConfigProviders)
    external = providers.Container(
        MockExternalServiceProviders,
        config=config,
    )
    repositories = providers.Container(
        MockRepositoryProviders,
        config=config,
        external_services=external,
    )
    services = providers.Container(
        MockServiceProviders,
        config=config,
        repositories=repositories,
        external_services=external,
    )

    # Convenience shortcuts
    settings = config.settings
    llm_service = services.llm_service
    embedding_service = external.sentence_transformer_service
    weaviate_client = external.weaviate_client
    document_repository = repositories.document_repository


def create_test_container() -> TestFitviseContainer:
    """Create a test container with all mock dependencies.
    
    Returns:
        TestFitviseContainer instance with mocked providers
    """
    return TestFitviseContainer()


def create_integration_test_container() -> FitviseContainer:
    """Create a container for integration tests with real dependencies.
    
    This container uses real configurations but isolated test environments.
    
    Returns:
        FitviseContainer configured for integration testing
    """
    from app.core.settings import Settings
    
    # Override settings for testing
    test_settings = Settings(
        app_name="FitVise Integration Test",
        app_version="1.0.0-integration",
        app_description="Integration test environment",
        environment="test",
        debug=True,
        database_url="sqlite:///test_integration.db",
        vector_store_path="/tmp/test_integration_vectors",
        upload_directory="/tmp/test_integration_uploads",
        knowledge_base_path="/tmp/test_integration_knowledge",
        # Use real services but test configuration
        llm_base_url="http://localhost:11434",
        llm_model="llama3.2:3b",
        embedding_model="Alibaba-NLP/gte-multilingual-base",
        # Test-specific Weaviate configuration
        weaviate_host="localhost",
        weaviate_port=8081,  # Different port for tests
        weaviate_scheme="http",
    )
    
    # Create container with test settings
    container = FitviseContainer()
    container.config.settings.override(providers.Singleton(lambda: test_settings))
    
    return container


class TestOverrides:
    """Utilities for overriding specific dependencies in tests."""
    
    @staticmethod
    def with_mock_llm_service(container: FitviseContainer, mock_response: str = "Test LLM response"):
        """Override LLM service with mock.
        
        Args:
            container: DI container to override
            mock_response: Mock response string
            
        Returns:
            Overridden container
        """
        mock_service = MagicMock(
            initialize=AsyncMock(),
            generate_response=AsyncMock(return_value=mock_response),
            health_check=AsyncMock(return_value=True),
        )
        
        container.services.llm_service.override(providers.Singleton(lambda: mock_service))
        return container
    
    @staticmethod
    def with_mock_embedding_service(container: FitviseContainer, mock_embedding: list = None):
        """Override embedding service with mock.
        
        Args:
            container: DI container to override
            mock_embedding: Mock embedding vector
            
        Returns:
            Overridden container
        """
        if mock_embedding is None:
            mock_embedding = [0.1, 0.2, 0.3]
        
        mock_service = MagicMock(
            is_loaded=True,
            initialize=AsyncMock(),
            embed_text=AsyncMock(return_value=mock_embedding),
            embed_batch=AsyncMock(return_value=[mock_embedding]),
            health_check=AsyncMock(return_value=True),
        )
        
        container.external.sentence_transformer_service.override(
            providers.Singleton(lambda: mock_service)
        )
        return container
    
    @staticmethod
    def with_mock_weaviate_client(container: FitviseContainer):
        """Override Weaviate client with mock.
        
        Args:
            container: DI container to override
            
        Returns:
            Overridden container
        """
        mock_client = MagicMock(
            is_connected=True,
            connect=AsyncMock(),
            health_check=AsyncMock(return_value=True),
        )
        
        container.external.weaviate_client.override(providers.Singleton(lambda: mock_client))
        return container


# Pytest fixtures
@pytest.fixture
def test_container():
    """Pytest fixture providing test container."""
    return create_test_container()


@pytest.fixture
def integration_test_container():
    """Pytest fixture providing integration test container."""
    return create_integration_test_container()


@pytest.fixture
def mock_llm_service(test_container):
    """Pytest fixture providing mock LLM service."""
    return test_container.services.llm_service()


@pytest.fixture
def mock_embedding_service(test_container):
    """Pytest fixture providing mock embedding service."""
    return test_container.external.sentence_transformer_service()


@pytest.fixture
def mock_weaviate_client(test_container):
    """Pytest fixture providing mock Weaviate client."""
    return test_container.external.weaviate_client()


# Export test utilities
__all__ = [
    "TestFitviseContainer",
    "create_test_container",
    "create_integration_test_container",
    "TestOverrides",
    # Pytest fixtures
    "test_container",
    "integration_test_container",
    "mock_llm_service",
    "mock_embedding_service",
    "mock_weaviate_client",
]
