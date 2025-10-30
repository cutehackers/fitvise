#!/usr/bin/env python3
"""
RAG Embedding Pipeline (Phase 3)

Phase 3 of RAG build pipeline. Takes processed documents from Phase 2,
generates embeddings using the configured model, and stores them in Weaviate.

Usage:
    python backend/scripts/embed_pipeline.py --config rag_pipeline.yaml [--verbose] [--batch-size 32]
"""

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, UUID
from uuid import uuid4

# Add backend/ to sys.path so "app" imports resolve when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline.config import PipelineSpec
from app.application.use_cases.indexing.build_ingestion_pipeline import (
    BuildIngestionPipelineUseCase,
    BuildIngestionPipelineRequest,
    BuildIngestionPipelineResponse,
)
from app.infrastructure.repositories.in_memory_document_repository import InMemoryDocumentRepository
from app.application.use_cases.chunking.semantic_chunking import SemanticChunkingUseCase
from app.application.use_cases.embedding.embed_document_chunks import EmbedDocumentChunksUseCase
from app.config.ml_models import get_chunking_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding pipeline execution."""

    success: bool
    documents_processed: int
    total_chunks: int
    unique_chunks: int
    duplicates_removed: int
    embeddings_generated: int
    embeddings_stored: int
    processing_time_seconds: float
    deduplication_stats: Optional[Dict[str, Any]] = None
    processing_stats: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "documents_processed": self.documents_processed,
            "total_chunks": self.total_chunks,
            "unique_chunks": self.unique_chunks,
            "duplicates_removed": self.duplicates_removed,
            "duplicates_percentage": round((self.duplicates_removed / self.total_chunks * 100), 2) if self.total_chunks > 0 else 0,
            "embeddings_generated": self.embeddings_generated,
            "embeddings_stored": self.embeddings_stored,
            "processing_time_seconds": round(self.processing_time_seconds, 2),
            "deduplication_stats": self.deduplication_stats,
            "processing_stats": self.processing_stats,
            "errors": self.errors,
            "warnings": self.warnings,
            "chunks_per_document": round(self.total_chunks / self.documents_processed, 2) if self.documents_processed > 0 else 0,
            "embedding_success_rate": round((self.embeddings_stored / self.unique_chunks * 100), 2) if self.unique_chunks > 0 else 0,
        }


class EmbeddingPipelineError(Exception):
    """Embedding pipeline execution failed."""
    pass


async def get_processed_document_ids(
    document_repository: InMemoryDocumentRepository,
    limit: Optional[int] = None
) -> List[UUID]:
    """Get IDs of processed documents that need embedding generation.

    Args:
        document_repository: Shared document repository instance
        limit: Optional limit on number of documents to process

    Returns:
        List of document IDs
    """
    try:
        # Get all documents from the shared repository
        documents = document_repository.list_all()

        # Convert to document IDs (limit if specified)
        document_ids = [doc.id for doc in documents[:limit]] if limit else [doc.id for doc in documents]

        if not document_ids:
            logger.warning("No processed documents found for embedding generation")

        logger.info(f"Found {len(document_ids)} processed documents for embedding")
        return document_ids

    except Exception as e:
        logger.error(f"Failed to get processed document IDs: {str(e)}")
        raise EmbeddingPipelineError(f"Document retrieval failed: {str(e)}")


async def run_embedding(
    config: PipelineSpec,
    document_repository: InMemoryDocumentRepository,
    batch_size: int = 32,
    deduplication_enabled: bool = True,
    max_retries: int = 3,
    show_progress: bool = True,
    document_limit: Optional[int] = None,
    verbose: bool = False
) -> EmbeddingResult:
    """Run the complete embedding pipeline.

    Args:
        config: Pipeline configuration
        document_repository: Shared document repository instance from orchestrator
        batch_size: Batch size for embedding generation
        deduplication_enabled: Whether to enable chunk deduplication
        max_retries: Maximum retry attempts for failed operations
        show_progress: Whether to show progress during processing
        document_limit: Optional limit on documents to process
        verbose: Enable verbose logging

    Returns:
        EmbeddingResult with comprehensive statistics

    Raises:
        EmbeddingPipelineError: If pipeline execution fails
    """
    import time
    start_time = time.time()

    logger.info("Starting RAG embedding pipeline...")
    if verbose:
        logger.info(f"Configuration: batch_size={batch_size}, deduplication={deduplication_enabled}, max_retries={max_retries}")

    try:
        # Initialize use cases
        logger.info("Initializing use cases...")

        # Document repository
        document_repository = InMemoryDocumentRepository()

        # Chunking use case
        chunking_use_case = SemanticChunkingUseCase(document_repository=document_repository)

        # Embedding use case
        embedding_use_case = EmbedDocumentChunksUseCase()

        # Build ingestion pipeline use case
        build_use_case = BuildIngestionPipelineUseCase(
            document_repository=document_repository,
            chunking_use_case=chunking_use_case,
            embedding_use_case=embedding_use_case
        )

        # Get document IDs to process from shared repository
        document_ids = await get_processed_document_ids(
            document_repository=document_repository,
            limit=document_limit
        )

        if not document_ids:
            logger.warning("No documents to process")
            return EmbeddingResult(
                success=True,
                documents_processed=0,
                total_chunks=0,
                unique_chunks=0,
                duplicates_removed=0,
                embeddings_generated=0,
                embeddings_stored=0,
                processing_time_seconds=time.time() - start_time,
                warnings=["No documents found for embedding generation"]
            )

        # Create pipeline request
        pipeline_request = BuildIngestionPipelineRequest(
            document_ids=document_ids,
            batch_size=batch_size,
            deduplication_enabled=deduplication_enabled,
            max_retries=max_retries,
            show_progress=show_progress,
            replace_existing_embeddings=False  # Incremental approach
        )

        logger.info(f"Processing {len(document_ids)} documents...")

        # Execute pipeline
        pipeline_response = await build_use_case.execute(pipeline_request)

        processing_time = time.time() - start_time

        # Build result
        result = EmbeddingResult(
            success=pipeline_response.success,
            documents_processed=len(document_ids),
            total_chunks=pipeline_response.deduplication_stats.total_chunks if pipeline_response.deduplication_stats else 0,
            unique_chunks=pipeline_response.deduplication_stats.unique_chunks if pipeline_response.deduplication_stats else 0,
            duplicates_removed=pipeline_response.deduplication_stats.duplicates_removed if pipeline_response.deduplication_stats else 0,
            embeddings_generated=pipeline_response.results.get("embeddings_generated", 0),
            embeddings_stored=pipeline_response.results.get("embeddings_stored", 0),
            processing_time_seconds=processing_time,
            deduplication_stats=pipeline_response.deduplication_stats.as_dict() if pipeline_response.deduplication_stats else None,
            processing_stats=pipeline_response.processing_stats,
            errors=pipeline_response.errors,
            warnings=[]
        )

        if pipeline_response.success:
            logger.info("🎉 Embedding pipeline completed successfully!")
            logger.info(f"Documents: {result.documents_processed}")
            logger.info(f"Chunks: {result.total_chunks} total, {result.unique_chunks} unique")
            logger.info(f"Duplicates removed: {result.duplicates_removed}")
            logger.info(f"Embeddings: {result.embeddings_generated} generated, {result.embeddings_stored} stored")
            logger.info(f"Processing time: {result.processing_time_seconds:.2f}s")
        else:
            logger.error("❌ Embedding pipeline failed")
            for error in pipeline_response.errors:
                logger.error(f"   - {error}")

        return result

    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Embedding pipeline failed: {str(e)}"
        logger.error(error_msg)

        return EmbeddingResult(
            success=False,
            documents_processed=0,
            total_chunks=0,
            unique_chunks=0,
            duplicates_removed=0,
            embeddings_generated=0,
            embeddings_stored=0,
            processing_time_seconds=processing_time,
            errors=[error_msg]
        )


async def main() -> int:
    """Main function to run embedding pipeline."""
    parser = argparse.ArgumentParser(description="Run RAG Phase 3 embedding pipeline")
    parser.add_argument("--config", required=True, help="Path to rag_pipeline.yaml (or .json)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding generation")
    parser.add_argument("--no-deduplication", action="store_true", help="Disable chunk deduplication")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retry attempts")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress display")
    parser.add_argument("--document-limit", type=int, help="Limit number of documents to process")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--output", help="Path to save embedding results as JSON")

    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Load pipeline configuration
        spec = PipelineSpec.from_file(args.config)

        # Run embedding pipeline
        result = await run_embedding(
            config=spec,
            batch_size=args.batch_size,
            deduplication_enabled=not args.no_deduplication,
            max_retries=args.max_retries,
            show_progress=not args.no_progress,
            document_limit=args.document_limit,
            verbose=args.verbose
        )

        # Save results if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(result.as_dict(), f, indent=2)
            print(f"Embedding results saved to {args.output}")

        # Print summary
        print("\n" + "="*60)
        print("RAG EMBEDDING PIPELINE SUMMARY")
        print("="*60)
        print(f"Success: {'✅ YES' if result.success else '❌ NO'}")
        print(f"Documents Processed: {result.documents_processed}")
        print(f"Total Chunks: {result.total_chunks}")
        print(f"Unique Chunks: {result.unique_chunks}")
        print(f"Duplicates Removed: {result.duplicates_removed} ({result.as_dict()['duplicates_percentage']}%)")
        print(f"Embeddings Generated: {result.embeddings_generated}")
        print(f"Embeddings Stored: {result.embeddings_stored}")
        print(f"Success Rate: {result.as_dict()['embedding_success_rate']}%")
        print(f"Avg Chunks per Document: {result.as_dict()['chunks_per_document']}")
        print(f"Processing Time: {result.processing_time_seconds:.2f}s")

        if result.warnings:
            print(f"\n⚠️  {len(result.warnings)} warnings:")
            for warning in result.warnings:
                print(f"   - {warning}")

        if result.errors:
            print(f"\n❌ {len(result.errors)} errors occurred:")
            for error in result.errors:
                print(f"   - {error}")
            return 1
        else:
            print("\n🎉 Embedding pipeline completed successfully!")
            return 0

    except Exception as e:
        logger.error(f"Embedding pipeline failed: {str(e)}")
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))