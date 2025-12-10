"""Examples of using RepositoryContainer in standalone scripts.

This demonstrates how to use the RepositoryContainer pattern outside of
FastAPI endpoints, such as in maintenance scripts, data migration scripts,
or background jobs.
"""
import asyncio
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from app.core.settings import Settings
from app.domain.entities.document import Document
from app.domain.value_objects.document_metadata import DocumentMetadata, DocumentFormat
from app.infrastructure.repositories.container import RepositoryContainer
from app.infrastructure.database.database import AsyncSessionLocal


# ==================== Example 1: In-Memory Mode ====================
async def example_in_memory():
    """Example: Using container with in-memory repositories (no database).

    This is useful for:
    - Quick prototyping
    - Testing scripts
    - Temporary data processing
    """
    print("=" * 60)
    print("Example 1: In-Memory Mode")
    print("=" * 60)

    # Override settings for in-memory mode
    settings = Settings()
    settings.database_url = "sqlite:///:memory:"

    # Create container without session for in-memory mode
    container = RepositoryContainer(settings)

    # Access repositories directly
    doc_repo = container.document_repository
    src_repo = container.data_source_repository

    # Create and save a test document
    metadata = DocumentMetadata(
        file_name="test.txt",
        file_path="/tmp/test.txt",
        file_size=1024,
        format=DocumentFormat.TXT,
    )

    document = Document(
        source_id=uuid4(),
        metadata=metadata,
        content="This is test content",
    )

    saved = await doc_repo.save(document)
    print(f"✓ Saved document: {saved.id}")

    # Retrieve the document
    found = await doc_repo.find_by_id(saved.id)
    print(f"✓ Retrieved document: {found.metadata.file_name if found else 'Not found'}")

    # List all documents
    all_docs = await doc_repo.find_all()
    print(f"✓ Total documents: {len(all_docs)}")

    print("\n")


# ==================== Example 2: Database Mode ====================
async def example_database():
    """Example: Using container with database repositories.

    This is useful for:
    - Production scripts
    - Data migrations
    - Batch processing with persistence
    """
    print("=" * 60)
    print("Example 2: Database Mode")
    print("=" * 60)

    settings = Settings()

    # Use async context manager for session
    async with AsyncSessionLocal() as session:
        # Create container with session
        container = RepositoryContainer(settings, session)

        # Access repositories
        doc_repo = container.document_repository

        # Create test document
        metadata = DocumentMetadata(
            file_name="database_test.pdf",
            file_path="/data/test.pdf",
            file_size=2048,
            format=DocumentFormat.PDF,
        )

        document = Document(
            source_id=uuid4(),
            metadata=metadata,
            content="Database-persisted content",
        )

        saved = await doc_repo.save(document)
        await session.commit()  # Commit transaction

        print(f"✓ Saved to database: {saved.id}")

        # Query documents
        all_docs = await doc_repo.find_all()
        print(f"✓ Total documents in database: {len(all_docs)}")

    print("\n")


# ==================== Example 3: Cleanup Script ====================
async def example_cleanup_old_documents():
    """Example: Maintenance script to delete old documents.

    This demonstrates a real-world use case for the container pattern.
    """
    print("=" * 60)
    print("Example 3: Cleanup Old Documents")
    print("=" * 60)

    settings = Settings()

    async with AsyncSessionLocal() as session:
        container = RepositoryContainer(settings, session)
        doc_repo = container.document_repository

        # Find documents older than 30 days
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)

        # Note: This assumes your repository has this method
        # If not, you'd need to implement it
        all_docs = await doc_repo.find_all()
        old_docs = [
            doc for doc in all_docs
            if doc.created_at and doc.created_at < cutoff_date
        ]

        print(f"Found {len(old_docs)} documents older than 30 days")

        # Delete old documents
        for doc in old_docs:
            await doc_repo.delete(doc.id)
            print(f"✓ Deleted: {doc.metadata.file_name}")

        await session.commit()
        print(f"\nCleaned up {len(old_docs)} old documents")

    print("\n")


# ==================== Example 4: Pipeline Usage ====================
async def example_pipeline():
    """Example: Using container in a pipeline.

    This shows how to create a container once and share it across
    multiple pipeline phases.
    """
    print("=" * 60)
    print("Example 4: Pipeline Usage")
    print("=" * 60)

    settings = Settings()

    async with AsyncSessionLocal() as session:
        # Create container once for entire pipeline
        container = RepositoryContainer(settings, session)

        # Simulate multiple pipeline phases using same container
        print("Phase 1: Ingestion")
        ingestion_repo = container.document_repository

        doc1 = Document(
            source_id=uuid4(),
            metadata=DocumentMetadata(
                file_name="phase1.txt",
                file_path="/data/phase1.txt",
                file_size=512,
                format=DocumentFormat.TXT,
            ),
            content="Phase 1 content",
        )

        await ingestion_repo.save(doc1)
        print(f"✓ Ingested: {doc1.metadata.file_name}")

        print("\nPhase 2: Processing")
        processing_repo = container.document_repository

        # Same instance as ingestion_repo!
        assert processing_repo is ingestion_repo

        docs = await processing_repo.find_all()
        for doc in docs:
            print(f"✓ Processing: {doc.metadata.file_name}")

        print("\nPhase 3: Validation")
        validation_repo = container.document_repository

        # All phases share the same repository instance
        assert validation_repo is ingestion_repo

        await session.commit()
        print("\n✓ Pipeline completed with shared repository")

    print("\n")


# ==================== Example 5: FastAPI Alternative ====================
async def example_fastapi_alternative():
    """Example: Creating container for FastAPI-like usage.

    This shows what FastAPI dependency injection does under the hood.
    """
    print("=" * 60)
    print("Example 5: FastAPI Alternative (Manual DI)")
    print("=" * 60)

    settings = Settings()

    # Simulate FastAPI request handling
    async def handle_request():
        """Simulates a FastAPI endpoint handler."""
        async with AsyncSessionLocal() as session:
            # This is what get_repository_container() does
            container = RepositoryContainer(settings, session)

            # This is what get_document_repository() does
            doc_repo = container.document_repository

            # Use repository in endpoint logic
            docs = await doc_repo.find_all()
            print(f"✓ Found {len(docs)} documents")

            return {"count": len(docs)}

    # Simulate multiple requests
    for i in range(3):
        print(f"\nRequest {i + 1}:")
        result = await handle_request()
        print(f"Response: {result}")

    print("\n")


# ==================== Main Runner ====================
async def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "RepositoryContainer Usage Examples" + " " * 14 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")

    # Run examples
    await example_in_memory()

    # Uncomment to run database examples (requires database setup)
    # await example_database()
    # await example_cleanup_old_documents()
    # await example_pipeline()
    # await example_fastapi_alternative()

    print("✓ All examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
