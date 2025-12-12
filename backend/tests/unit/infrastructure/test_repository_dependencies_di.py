"""DI-backed repository dependency helpers."""

import pytest

from app.di import container
from app.di.testing import create_test_container
from app.infrastructure.persistence.repositories.dependencies import (
    get_data_source_repository,
    get_document_repository,
    get_embedding_repository,
    get_search_repository,
    get_transaction_session,
    repositories_health,
)


@pytest.mark.asyncio
async def test_repository_helpers_use_container_overrides():
    """Repository dependencies should return instances from the active container override."""
    test_container = create_test_container()
    with container.override(test_container):
        doc_repo = await get_document_repository()
        data_repo = await get_data_source_repository()
        embed_repo = await get_embedding_repository()
        search_repo = await get_search_repository()

    assert doc_repo is test_container.repositories.document_repository()
    assert data_repo is test_container.repositories.data_source_repository()
    assert embed_repo is test_container.repositories.embedding_repository()
    assert search_repo is test_container.repositories.search_repository()


@pytest.mark.asyncio
async def test_transaction_session_yields_session():
    """Transaction session dependency should yield a session object."""
    test_container = create_test_container()
    with container.override(test_container):
        agen = get_transaction_session()
        session = await agen.__anext__()
        assert session is not None
        with pytest.raises(StopAsyncIteration):
            await agen.__anext__()


@pytest.mark.asyncio
async def test_repositories_health_uses_provider():
    """Health check helper should delegate to container provider."""
    test_container = create_test_container()
    with container.override(test_container):
        health = await repositories_health()

    assert health["overall"] is True
    assert health["database"] is True
    assert health["weaviate"] is True
