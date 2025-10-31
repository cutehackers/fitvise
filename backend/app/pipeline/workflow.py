"""RAG Workflow Orchestrator.

This module provides a unified workflow class that orchestrates all phases
of the RAG pipeline with proper dependency injection and shared repository state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.pipeline.config import PipelineSpec
from app.pipeline.phases import InfrastructurePhase, IngestionPhase, EmbeddingPhase
from app.pipeline.phases.infrastructure_phase import InfrastructureValidationResult
from app.pipeline.phases.embedding_phase import EmbeddingResult
from app.pipeline.contracts import RunSummary
from app.domain.repositories.document_repository import DocumentRepository
from app.domain.repositories.data_source_repository import DataSourceRepository
from app.infrastructure.repositories.in_memory_document_repository import (
    InMemoryDocumentRepository,
)
from app.infrastructure.repositories.in_memory_data_source_repository import (
    InMemoryDataSourceRepository,
)
from scripts.rag_summary import (
    RagIngestionSummary,
    create_infrastructure_phase_result,
    create_ingestion_phase_result,
    create_embedding_phase_result,
    InfrastructureResults,
    IngestionResults,
    EmbeddingResults,
)

logger = logging.getLogger(__name__)


@dataclass
class RepositoryBundle:
    """Bundle of shared repository instances for pipeline execution.

    This ensures all phases work with the same repository instances,
    maintaining data continuity throughout the pipeline.
    """

    document_repository: DocumentRepository
    data_source_repository: DataSourceRepository


class RAGWorkflow:
    """Orchestrates the complete RAG build pipeline with dependency injection.

    This class coordinates all three phases of the RAG pipeline:
    1. Infrastructure validation
    2. Document ingestion and processing
    3. Embedding generation and storage

    All phases share the same repository instances to ensure data continuity.
    """

    def __init__(
        self,
        repositories: Optional[RepositoryBundle] = None,
        verbose: bool = False,
    ):
        """Initialize the RAG workflow orchestrator.

        Args:
            repositories: Optional pre-configured repository bundle.
                         If not provided, creates in-memory repositories.
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        # Initialize or use provided repositories
        if repositories is None:
            self.repositories = RepositoryBundle(
                document_repository=InMemoryDocumentRepository(),
                data_source_repository=InMemoryDataSourceRepository(),
            )
        else:
            self.repositories = repositories

        # Initialize phases with shared repositories
        self.infrastructure_phase = InfrastructurePhase(verbose=verbose)
        self.ingestion_phase = IngestionPhase(
            document_repository=self.repositories.document_repository,
            data_source_repository=self.repositories.data_source_repository,
            verbose=verbose,
        )
        self.embedding_phase = EmbeddingPhase(
            document_repository=self.repositories.document_repository,
            verbose=verbose,
        )

        # Summary tracking
        self.summary = RagIngestionSummary()
        self.start_time: Optional[datetime] = None

    async def run_infrastructure_check(
        self, spec: PipelineSpec
    ) -> InfrastructureValidationResult:
        """Run Phase 1: Infrastructure validation independently.

        Args:
            spec: Pipeline specification

        Returns:
            InfrastructureValidationResult

        Raises:
            InfrastructureValidationError: If critical validation fails
        """
        logger.info("🔧 Phase 1: Infrastructure Setup and Validation")
        phase_start = datetime.now(timezone.utc)

        try:
            result = await self.infrastructure_phase.execute(spec)
            phase_end = datetime.now(timezone.utc)
            execution_time = (phase_end - phase_start).total_seconds()

            logger.info(
                f"✅ Phase 1 completed in {execution_time:.2f}s"
            )
            return result

        except Exception as e:
            phase_end = datetime.now(timezone.utc)
            execution_time = (phase_end - phase_start).total_seconds()
            logger.error(
                f"❌ Phase 1 failed after {execution_time:.2f}s: {str(e)}"
            )
            raise

    async def run_ingestion(
        self, spec: PipelineSpec, dry_run: bool = False
    ) -> RunSummary:
        """Run Phase 2: Document ingestion independently.

        Args:
            spec: Pipeline specification
            dry_run: Run in dry-run mode (skip actual storage)

        Returns:
            RunSummary with processing results
        """
        logger.info("📄 Phase 2: Document Ingestion and Processing")
        phase_start = datetime.now(timezone.utc)

        try:
            # Override storage for dry run
            if dry_run:
                data = spec.model_dump()
                data.setdefault("storage", {})
                data["storage"]["bucket"] = "rag-dry-run"
                data["storage"]["provider"] = "local"
                spec = PipelineSpec.model_validate(data)

            result = await self.ingestion_phase.execute(spec, dry_run=dry_run)
            phase_end = datetime.now(timezone.utc)
            execution_time = (phase_end - phase_start).total_seconds()

            logger.info(
                f"✅ Phase 2 completed in {execution_time:.2f}s: {result.processed} documents processed"
            )
            return result

        except Exception as e:
            phase_end = datetime.now(timezone.utc)
            execution_time = (phase_end - phase_start).total_seconds()
            logger.error(
                f"❌ Phase 2 failed after {execution_time:.2f}s: {str(e)}"
            )
            raise

    async def run_embedding(
        self,
        spec: PipelineSpec,
        batch_size: int = 32,
        document_limit: Optional[int] = None,
    ) -> EmbeddingResult:
        """Run Phase 3: Embedding generation independently.

        Args:
            spec: Pipeline specification
            batch_size: Batch size for embedding generation
            document_limit: Optional limit on documents to process

        Returns:
            EmbeddingResult with comprehensive statistics
        """
        logger.info("🔢 Phase 3: Embedding Generation and Storage")
        phase_start = datetime.now(timezone.utc)

        try:
            result = await self.embedding_phase.execute(
                spec=spec,
                batch_size=batch_size,
                deduplication_enabled=True,
                max_retries=3,
                show_progress=not self.verbose,
                document_limit=document_limit,
            )
            phase_end = datetime.now(timezone.utc)
            execution_time = (phase_end - phase_start).total_seconds()

            logger.info(
                f"✅ Phase 3 completed in {execution_time:.2f}s: {result.embeddings_stored} embeddings stored"
            )
            return result

        except Exception as e:
            phase_end = datetime.now(timezone.utc)
            execution_time = (phase_end - phase_start).total_seconds()
            logger.error(
                f"❌ Phase 3 failed after {execution_time:.2f}s: {str(e)}"
            )
            raise

    async def run_complete_pipeline(
        self,
        spec: PipelineSpec,
        dry_run: bool = False,
        batch_size: int = 32,
        document_limit: Optional[int] = None,
        output_dir: Optional[str] = None,
    ) -> RagIngestionSummary:
        """Run the complete RAG pipeline (all 3 phases sequentially).

        Args:
            spec: Pipeline specification
            dry_run: Run in dry-run mode (skip actual storage)
            batch_size: Batch size for embedding generation
            document_limit: Optional limit on documents to process
            output_dir: Directory to save reports and outputs

        Returns:
            RagIngestionSummary with comprehensive results from all phases
        """
        self.start_time = datetime.now(timezone.utc)
        self.summary.mark_started()

        logger.info("🚀 Starting RAG Build Pipeline")
        logger.info(f"Configuration: {spec}")
        logger.info(f"Output Directory: {output_dir or 'default'}")
        logger.info(f"Dry Run: {dry_run}")

        # Create output directory if specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        try:
            # Phase 1: Infrastructure Validation
            await self._run_infrastructure_with_tracking(spec, output_dir)

            # Phase 2: Document Ingestion
            await self._run_ingestion_with_tracking(spec, dry_run, output_dir)

            # Phase 3: Embedding Generation
            await self._run_embedding_with_tracking(
                spec, batch_size, document_limit, output_dir
            )

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

            # Save error report
            if output_dir:
                await self._save_reports(output_dir)

            raise

    async def _run_infrastructure_with_tracking(
        self, spec: PipelineSpec, output_dir: Optional[str]
    ) -> None:
        """Run infrastructure phase with result tracking."""
        phase_start = datetime.now(timezone.utc)

        try:
            validation_result = await self.infrastructure_phase.execute(spec)
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
                end_time=phase_end,
            )

            # Create infrastructure-specific results
            infra_results = InfrastructureResults(
                embedding_service_status=validation_result.validation_results.get(
                    "embedding_service", {}
                ),
                weaviate_status=validation_result.validation_results.get(
                    "weaviate_schema", {}
                ),
                object_storage_status=validation_result.validation_results.get(
                    "object_storage", {}
                ),
                configuration_status=validation_result.validation_results.get(
                    "configuration", {}
                ),
            )

            self.summary.add_phase_result(phase_result)
            self.summary.set_infrastructure_results(infra_results)

            # Save phase-specific report
            if output_dir:
                self._save_phase_report(
                    output_dir, "phase_1_infrastructure.json", phase_result.as_dict()
                )

            logger.info("✅ Phase 1 completed successfully")

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
                end_time=phase_end,
            )

            self.summary.add_phase_result(phase_result)
            raise

    async def _run_ingestion_with_tracking(
        self, spec: PipelineSpec, dry_run: bool, output_dir: Optional[str]
    ) -> None:
        """Run ingestion phase with result tracking."""
        phase_start = datetime.now(timezone.utc)

        try:
            ingestion_summary = await self.ingestion_phase.execute(spec, dry_run)
            phase_end = datetime.now(timezone.utc)
            execution_time = (phase_end - phase_start).total_seconds()

            # Create phase result
            phase_result = create_ingestion_phase_result(
                success=ingestion_summary.processed > 0,
                execution_time=execution_time,
                pipeline_summary=ingestion_summary.as_dict(),
                errors=[
                    str(error.get("message", error))
                    for error in ingestion_summary.errors
                ],
                start_time=phase_start,
                end_time=phase_end,
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
                processing_errors=[str(error) for error in ingestion_summary.errors],
            )

            self.summary.add_phase_result(phase_result)
            self.summary.set_ingestion_results(ingestion_results)

            # Save phase-specific reports
            if output_dir:
                self._save_phase_report(
                    output_dir, "phase_2_ingestion.json", phase_result.as_dict()
                )
                self._save_phase_report(
                    output_dir,
                    "ingestion_detailed.json",
                    ingestion_summary.as_dict(),
                )

            if ingestion_summary.processed > 0:
                logger.info(
                    f"✅ Phase 2 completed: {ingestion_summary.processed} documents processed"
                )
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
                end_time=phase_end,
            )

            self.summary.add_phase_result(phase_result)
            raise

    async def _run_embedding_with_tracking(
        self,
        spec: PipelineSpec,
        batch_size: int,
        document_limit: Optional[int],
        output_dir: Optional[str],
    ) -> None:
        """Run embedding phase with result tracking."""
        phase_start = datetime.now(timezone.utc)

        try:
            embedding_result = await self.embedding_phase.execute(
                spec=spec,
                batch_size=batch_size,
                deduplication_enabled=True,
                max_retries=3,
                show_progress=not self.verbose,
                document_limit=document_limit,
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
                end_time=phase_end,
            )

            # Create embedding-specific results
            embedding_results = EmbeddingResults(
                documents_processed=embedding_result.documents_processed,
                total_chunks=embedding_result.total_chunks,
                unique_chunks=embedding_result.unique_chunks,
                duplicates_removed=embedding_result.duplicates_removed,
                embeddings_generated=embedding_result.embeddings_generated,
                embeddings_stored=embedding_result.embeddings_stored,
                deduplication_stats=embedding_result.deduplication_stats,
            )

            self.summary.add_phase_result(phase_result)
            self.summary.set_embedding_results(embedding_results)

            # Save phase-specific reports
            if output_dir:
                self._save_phase_report(
                    output_dir, "phase_3_embedding.json", phase_result.as_dict()
                )
                self._save_phase_report(
                    output_dir, "embedding_detailed.json", embedding_result.as_dict()
                )

            if embedding_result.success:
                logger.info(
                    f"✅ Phase 3 completed: {embedding_result.embeddings_stored} embeddings stored"
                )
            else:
                logger.error("❌ Phase 3 failed")
                raise Exception(
                    f"Embedding pipeline failed with {len(embedding_result.errors)} errors"
                )

        except Exception as e:
            phase_end = datetime.now(timezone.utc)
            execution_time = (phase_end - phase_start).total_seconds()

            phase_result = create_embedding_phase_result(
                success=False,
                execution_time=execution_time,
                embedding_result={},
                errors=[str(e)],
                start_time=phase_start,
                end_time=phase_end,
            )

            self.summary.add_phase_result(phase_result)
            raise

    def _save_phase_report(
        self, output_dir: str, filename: str, data: dict
    ) -> None:
        """Save a phase-specific report to file."""
        import json

        output_path = Path(output_dir) / filename
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    async def _save_reports(self, output_dir: str) -> None:
        """Save all reports to the output directory."""
        output_path = Path(output_dir)

        # Save main summary
        main_report_path = output_path / "rag_ingestion_summary.json"
        self.summary.save_to_file(str(main_report_path))

        # Save human-readable report
        readable_report_path = output_path / "rag_ingestion_report.txt"
        with open(readable_report_path, "w") as f:
            f.write(f"RAG Ingestion Pipeline Report\n")
            f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            f.write(self.summary.as_json(indent=2))

        logger.info(f"📊 Reports saved to {output_dir}")
        logger.info(f"   - Main summary: {main_report_path}")
        logger.info(f"   - Detailed report: {readable_report_path}")
