"""DI wiring tests for RAGWorkflow."""

import pytest

from app.di import container as global_container
from app.di.testing import create_test_container
from app.pipeline.workflow import PipelineServices, RAGWorkflow, RepositoryBundle


def test_pipeline_workflow_from_test_container():
    """Ensure pipeline workflow from test container wires mocked dependencies."""
    test_container = create_test_container()
    workflow = test_container.services.pipeline_workflow()

    assert isinstance(workflow, RAGWorkflow)
    assert isinstance(workflow.repositories, RepositoryBundle)
    assert workflow.repositories.document_repository is test_container.repositories.document_repository()
    assert workflow.repositories.embedding_repository is test_container.repositories.embedding_repository()
    assert isinstance(workflow.services, PipelineServices)
    assert workflow.services.embedding_service is test_container.external.sentence_transformer_service()
    assert workflow.services.weaviate_client is test_container.external.weaviate_client()


@pytest.mark.asyncio
async def test_workflow_uses_di_override_for_defaults():
    """RAGWorkflow should pull dependencies from the active DI container when none are supplied."""
    test_container = create_test_container()
    with global_container.override(test_container):
        workflow = RAGWorkflow(verbose=False)

        assert workflow.repositories.document_repository is test_container.repositories.document_repository()
        assert workflow.services.embedding_service is test_container.external.sentence_transformer_service()
