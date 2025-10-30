#!/usr/bin/env python3
"""
RAG Build Pipeline Orchestrator

Main orchestrator script that coordinates all three phases of the RAG build pipeline:
1. Infrastructure Setup and Validation
2. Document Ingestion and Processing
3. Embedding Generation and Storage

Usage:
    python scripts/build_rag_pipeline.py --config rag_pipeline.yaml [--verbose] [--output-dir ./reports]
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Add backend/ to sys.path so "app" imports resolve when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import pipeline phases
from scripts.setup_rag_infrastructure import validate_infrastructure, InfrastructureValidationError
from scripts.ingestion import run_ingestion
from scripts.embedding import run_embedding
from scripts.rag_summary import (
    RagIngestionSummary,
    create_infrastructure_phase_result,
    create_ingestion_phase_result,
    create_embedding_phase_result,
    InfrastructureResults,
    IngestionResults,
    EmbeddingResults
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGPipelineOrchestrator:
    """Orchestrator for the complete RAG build pipeline."""

    def __init__(self, verbose: bool = False):
        """Initialize the orchestrator.

        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        self.summary = RagIngestionSummary()
        self.start_time: Optional[datetime] = None

        # Shared repositories for consistent data across phases
        self.document_repository = None
        self.data_source_repository = None

    async def run_pipeline(
        self,
        config_path: str,
        output_dir: Optional[str] = None,
        dry_run: bool = False,
        embedding_batch_size: int = 32,
        document_limit: Optional[int] = None
    ) -> RagIngestionSummary:
        """Run the complete RAG build pipeline.

        Args:
            config_path: Path to pipeline configuration file
            output_dir: Directory to save reports and outputs
            dry_run: Run in dry-run mode (skip actual storage)
            embedding_batch_size: Batch size for embedding generation
            document_limit: Limit number of documents to process

        Returns:
            RagIngestionSummary with comprehensive results
        """
        self.start_time = datetime.now(timezone.utc)
        self.summary.mark_started()

        logger.info("🚀 Starting RAG Build Pipeline")
        logger.info(f"Configuration: {config_path}")
        logger.info(f"Output Directory: {output_dir or 'default'}")
        logger.info(f"Dry Run: {dry_run}")

        # Create output directory if specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        try:
            # Initialize shared repositories
            self._init_shared_repositories()

            # Phase 1: Infrastructure Validation
            await self._run_infrastructure(config_path, output_dir)

            # Phase 2: Document Ingestion
            await self._run_ingestion(config_path, output_dir, dry_run)

            # Phase 3: Embedding Generation
            await self._run_embedding(config_path, output_dir, embedding_batch_size, document_limit)

            # Complete pipeline
            self.summary.mark_completed()

            # Save reports
            if output_dir:
                await self._save_reports(output_dir)

            return self.summary

        except Exception as e:
            logger.error(f"❌ RAG pipeline failed: {str(e)}")
            self.summary.success = False
            if not self.summary.failure_phase:
                self.summary.failure_phase = "Unknown"
            self.summary.mark_completed()

            # Save error report if output directory specified
            if output_dir:
                await self._save_reports(output_dir)

            raise

    def _init_shared_repositories(self) -> None:
        """Initialize shared repository instances for consistent data across phases."""
        from app.infrastructure.repositories.in_memory_document_repository import InMemoryDocumentRepository
        from app.infrastructure.repositories.in_memory_data_source_repository import InMemoryDataSourceRepository

        logger.info("🔧 Initializing shared repositories...")
        self.document_repository = InMemoryDocumentRepository()
        self.data_source_repository = InMemoryDataSourceRepository()
        logger.info("✅ Shared repositories initialized")

    async def _run_infrastructure(self, config_path: str, output_dir: Optional[str]) -> None:
        """Run Phase 1: Infrastructure Setup and Validation."""
        from app.pipeline.config import PipelineSpec

        logger.info("🔧 Phase 1: Infrastructure Setup and Validation")
        phase_start = datetime.now(timezone.utc)

        try:
            # Load configuration
            spec = PipelineSpec.from_file(config_path)

            # Run infrastructure validation
            validation_result = await validate_infrastructure(spec, verbose=self.verbose)

            phase_end = datetime.now(timezone.utc)
            execution_time = (phase_end - phase_start).total_seconds()

            # Create phase result
            phase_result = create_infrastructure_phase_result(
                success=validation_result.success,
                execution_time=execution_time,
                validation_results=validation_result.validation_results,
                errors=validation_result.errors,
                warnings=validation_result.warnings,
                start_time=phase_start,
                end_time=phase_end
            )

            # Create infrastructure-specific results
            infra_results = InfrastructureResults(
                embedding_service_status=validation_result.validation_results.get("embedding_service", {}),
                weaviate_status=validation_result.validation_results.get("weaviate_schema", {}),
                object_storage_status=validation_result.validation_results.get("object_storage", {}),
                configuration_status=validation_result.validation_results.get("configuration", {})
            )

            self.summary.add_phase_result(phase_result)
            self.summary.set_infrastructure_results(infra_results)

            # Save phase-specific report
            if output_dir:
                phase_report_path = Path(output_dir) / "phase_1_infrastructure.json"
                with open(phase_report_path, 'w') as f:
                    json.dump(phase_result.as_dict(), f, indent=2)

            if validation_result.success:
                logger.info("✅ Phase 1 completed successfully")
            else:
                logger.error("❌ Phase 1 failed")
                raise InfrastructureValidationError(
                    "Infrastructure validation failed",
                    validation_result.errors,
                    validation_result.validation_results
                )

        except Exception as e:
            phase_end = datetime.now(timezone.utc)
            execution_time = (phase_end - phase_start).total_seconds()

            phase_result = create_infrastructure_phase_result(
                success=False,
                execution_time=execution_time,
                validation_results={},
                errors=[str(e)],
                warnings=[],
                start_time=phase_start,
                end_time=phase_end
            )

            self.summary.add_phase_result(phase_result)
            raise

    async def _run_ingestion(self, config_path: str, output_dir: Optional[str], dry_run: bool) -> None:
        """Run Phase 2: Document Ingestion and Processing."""
        logger.info("📄 Phase 2: Document Ingestion and Processing")
        phase_start = datetime.now(timezone.utc)

        try:
            # Load configuration
            from app.pipeline.config import PipelineSpec
            spec = PipelineSpec.from_file(config_path)

            # Override storage for dry run
            if dry_run:
                data = spec.model_dump()
                data.setdefault("storage", {})
                data["storage"]["bucket"] = "rag-dry-run"
                data["storage"]["provider"] = "local"
                spec = PipelineSpec.model_validate(data)

            # Run ingestion with shared repository
            ingestion_summary = await run_ingestion(
                config=spec,
                document_repository=self.document_repository,
                dry_run=dry_run,
                verbose=self.verbose
            )

            phase_end = datetime.now(timezone.utc)
            execution_time = (phase_end - phase_start).total_seconds()

            # Create phase result
            phase_result = create_ingestion_phase_result(
                success=ingestion_summary.processed > 0,
                execution_time=execution_time,
                pipeline_summary=ingestion_summary.as_dict(),
                errors=[str(error.get('message', error)) for error in ingestion_summary.errors],
                start_time=phase_start,
                end_time=phase_end
            )

            # Create ingestion-specific results
            counters = ingestion_summary.counters
            ingestion_results = IngestionResults(
                documents_discovered=counters.get("discovered", 0),
                documents_processed=ingestion_summary.processed,
                documents_skipped=ingestion_summary.skipped,
                documents_failed=ingestion_summary.failed,
                chunks_generated=counters.get("chunking", {}).get("total_chunks", 0),
                storage_objects_created=len(ingestion_summary.stored),
                processing_errors=[str(error) for error in ingestion_summary.errors]
            )

            self.summary.add_phase_result(phase_result)
            self.summary.set_ingestion_results(ingestion_results)

            # Save phase-specific report
            if output_dir:
                phase_report_path = Path(output_dir) / "phase_2_ingestion.json"
                with open(phase_report_path, 'w') as f:
                    json.dump(phase_result.as_dict(), f, indent=2)

                # Also save the detailed ingestion summary
                ingestion_report_path = Path(output_dir) / "ingestion_detailed.json"
                with open(ingestion_report_path, 'w') as f:
                    json.dump(ingestion_summary.as_dict(), f, indent=2)

            if ingestion_summary.processed > 0:
                logger.info(f"✅ Phase 2 completed successfully: {ingestion_summary.processed} documents processed")
            else:
                logger.warning("⚠️ Phase 2 completed but no documents were processed")

        except Exception as e:
            phase_end = datetime.now(timezone.utc)
            execution_time = (phase_end - phase_start).total_seconds()

            phase_result = create_ingestion_phase_result(
                success=False,
                execution_time=execution_time,
                pipeline_summary={},
                errors=[str(e)],
                start_time=phase_start,
                end_time=phase_end
            )

            self.summary.add_phase_result(phase_result)
            raise

    async def _run_embedding(
        self,
        config_path: str,
        output_dir: Optional[str],
        batch_size: int,
        document_limit: Optional[int]
    ) -> None:
        """Run Phase 3: Embedding Generation and Storage."""
        logger.info("🔢 Phase 3: Embedding Generation and Storage")
        phase_start = datetime.now(timezone.utc)

        try:
            from app.pipeline.config import PipelineSpec

            spec = PipelineSpec.from_file(config_path)

            # Run embedding pipeline with shared repository
            embedding_result = await run_embedding(
                config=spec,
                document_repository=self.document_repository,  # Pass shared repository
                batch_size=batch_size,
                deduplication_enabled=True,
                max_retries=3,
                show_progress=not self.verbose,
                document_limit=document_limit,
                verbose=self.verbose
            )

            phase_end = datetime.now(timezone.utc)
            execution_time = (phase_end - phase_start).total_seconds()

            # Create phase result
            phase_result = create_embedding_phase_result(
                success=embedding_result.success,
                execution_time=execution_time,
                embedding_result=embedding_result.as_dict(),
                errors=embedding_result.errors,
                start_time=phase_start,
                end_time=phase_end
            )

            # Create embedding-specific results
            embedding_results = EmbeddingResults(
                documents_processed=embedding_result.documents_processed,
                total_chunks=embedding_result.total_chunks,
                unique_chunks=embedding_result.unique_chunks,
                duplicates_removed=embedding_result.duplicates_removed,
                embeddings_generated=embedding_result.embeddings_generated,
                embeddings_stored=embedding_result.embeddings_stored,
                deduplication_stats=embedding_result.deduplication_stats
            )

            self.summary.add_phase_result(phase_result)
            self.summary.set_embedding_results(embedding_results)

            # Save phase-specific report
            if output_dir:
                phase_report_path = Path(output_dir) / "phase_3_embedding.json"
                with open(phase_report_path, 'w') as f:
                    json.dump(phase_result.as_dict(), f, indent=2)

                # Also save detailed embedding results
                embedding_report_path = Path(output_dir) / "embedding_detailed.json"
                with open(embedding_report_path, 'w') as f:
                    json.dump(embedding_result.as_dict(), f, indent=2)

            if embedding_result.success:
                logger.info(f"✅ Phase 3 completed successfully: {embedding_result.embeddings_stored} embeddings stored")
            else:
                logger.error("❌ Phase 3 failed")
                raise Exception(f"Embedding pipeline failed with {len(embedding_result.errors)} errors")

        except Exception as e:
            phase_end = datetime.now(timezone.utc)
            execution_time = (phase_end - phase_start).total_seconds()

            phase_result = create_embedding_phase_result(
                success=False,
                execution_time=execution_time,
                embedding_result={},
                errors=[str(e)],
                start_time=phase_start,
                end_time=phase_end
            )

            self.summary.add_phase_result(phase_result)
            raise

    async def _save_reports(self, output_dir: str) -> None:
        """Save all reports to the output directory."""
        output_path = Path(output_dir)

        # Save main summary
        main_report_path = output_path / "rag_ingestion_summary.json"
        self.summary.save_to_file(str(main_report_path))

        # Save human-readable report
        readable_report_path = output_path / "rag_ingestion_report.txt"
        with open(readable_report_path, 'w') as f:
            f.write(f"RAG Ingestion Pipeline Report\n")
            f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            f.write(self.summary.as_json(indent=2))

        logger.info(f"📊 Reports saved to {output_dir}")
        logger.info(f"   - Main summary: {main_report_path}")
        logger.info(f"   - Detailed report: {readable_report_path}")


async def main() -> int:
    """Main function to run the RAG build pipeline."""
    parser = argparse.ArgumentParser(description="Build RAG ingestion pipeline")
    parser.add_argument("--config", required=True, help="Path to rag_pipeline.yaml (or .json)")
    parser.add_argument("--output-dir", help="Directory to save reports and outputs")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding generation")
    parser.add_argument("--document-limit", type=int, help="Limit number of documents to process")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop pipeline on first error (default)")

    args = parser.parse_args()

    try:
        # Initialize orchestrator
        orchestrator = RAGPipelineOrchestrator(verbose=args.verbose)

        # Run pipeline
        summary = await orchestrator.run_pipeline(
            config_path=args.config,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            embedding_batch_size=args.batch_size,
            document_limit=args.document_limit
        )

        # Print final summary
        summary.print_summary()

        # Return appropriate exit code
        return 0 if summary.success else 1

    except KeyboardInterrupt:
        logger.info("🛑 Pipeline interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))