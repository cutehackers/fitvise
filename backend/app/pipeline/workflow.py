"""RAG Workflow Orchestrator.

This module provides a unified workflow class that orchestrates all tasks
of the RAG pipeline with proper dependency injection and shared repository state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.pipeline.config import PipelineSpec
from app.pipeline.phases import RagInfrastructureTask, RagIngestionTask, RagEmbeddingTask
from app.pipeline.phases.infrastructure_task import (
    InfrastructureResult,
    RagInfrastructureTaskReport,
)
from app.pipeline.phases.embedding_task import (
    EmbeddingResult,
    RagEmbeddingTaskReport,
)
from app.pipeline.phases.ingestion_task import RagIngestionTaskReport
from app.pipeline.contracts import RunSummary
from app.domain.entities.chunk_load_policy import ChunkLoadPolicy
from app.domain.repositories.document_repository import DocumentRepository
from app.domain.repositories.data_source_repository import DataSourceRepository
from app.infrastructure.repositories.container import RepositoryContainer
from app.infrastructure.external_services import ExternalServicesContainer
from app.core.settings import Settings, get_settings
from scripts.rag_summary import (
    RagBuildSummary,
    create_infrastructure_phase_result,
    create_ingestion_phase_result,
    create_embedding_phase_result,
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

    This class coordinates all three tasks of the RAG pipeline:
    1. Infrastructure validation
    2. Document ingestion and processing
    3. Embedding generation and storage

    All tasks share the same repository instances to ensure data continuity.

    ## Pipeline Configuration Options

    The RAGWorkflow provides two key configuration options that significantly impact
    the pipeline flow and performance:

    ### enable_semantic_chunking (Task 2 - Document Ingestion)

    **Purpose**: Controls the chunking strategy used during document ingestion.

    **Impact on Pipeline Flow**:

    When **enable_semantic_chunking=False** (Default - Performance Mode):
    - Uses sentence-based splitting with fixed size boundaries
    - No embedding generation required during chunking
    - Faster processing, lower computational cost
    - Chunks are created based on sentence boundaries and size limits
    - Optimal for: Large document sets, processing speed prioritized, initial indexing

    When **enable_semantic_chunking=True** (Quality Mode):
    - Uses semantic chunking with embedding analysis
    - Generates temporary embeddings to determine semantic boundaries
    - Chunks are created where topic shifts occur (based on embedding similarity)
    - Higher computational cost but better semantic coherence
    - Optimal for: High-quality retrieval, document sets where context matters,
                  academic/technical content requiring coherent chunks

    **Performance Considerations**:
    - False: 2-3x faster processing, minimal resource usage
    - True: Slower processing due to embedding generation, higher memory/CPU usage

    ### Task Separation (Chunking in Task 2, Embedding in Task 3)

    **Architecture**: Clear separation of concerns between pipeline tasks.

    **Task 2 (Ingestion)**: Responsible for all chunking operations
    - Processes documents and generates chunks
    - Stores chunks in repository for Task 3 to consume
    - Uses semantic or sentence splitter based on `enable_semantic_chunking`

    **Task 3 (Embedding Generation)**: Consumes chunks from Task 2
    - Loads existing chunks from repository (created in Task 2)
    - Validates chunk availability before proceeding
    - Generates and stores embeddings for existing chunks only
    - Does NOT perform any chunking operations

    **Important**: If you need to change chunking strategy, re-run Task 2 (Ingestion) with
    new parameters. Task 3 will always use chunks created by Task 2.

    ### Recommended Configurations

    **Production Pipeline (Default)**:
    ```python
    await workflow.run_complete_pipeline(
        spec=spec,
        enable_semantic_chunking=False,  # Fast sentence-based chunking in Task 2
    )
    ```

    **High-Quality Pipeline**:
    ```python
    await workflow.run_complete_pipeline(
        spec=spec,
        enable_semantic_chunking=True,   # Semantic boundaries in Task 2
    )
    ```

    **Parameter Experimentation**:
    ```python
    # Run ingestion once with semantic chunking
    await workflow.run_ingestion(
        spec=spec,
        enable_semantic_chunking=True
    )

    # Test different embedding parameters using existing chunks
    await workflow.run_embedding(
        spec=spec,
        batch_size=64,  # Test different batch size
        enable_semantic_chunking=True  # Match Task 2 setting for fallback consistency
    )
    ```

    **Pipeline Debugging**:
    ```python
    # Re-run ingestion if chunking issues detected
    await workflow.run_ingestion(
        spec=spec,
        enable_semantic_chunking=False,  # Try different chunking strategy
        document_limit=10  # Test on subset first
    )

    # Then run embedding on new chunks (matching chunking strategy)
    await workflow.run_embedding(
        spec=spec,
        document_limit=10,
        enable_semantic_chunking=False  # Match Task 2 setting
    )
    ```

    These options provide fine-grained control over the quality-performance tradeoff
    and enable flexible pipeline execution patterns for different use cases.
    """

    def __init__(
        self,
        repositories: Optional[RepositoryBundle] = None,
        external_services: Optional[ExternalServicesContainer] = None,
        session: Optional[AsyncSession] = None,
        verbose: bool = False,
    ):
        """Initialize the RAG workflow orchestrator.

        Args:
            repositories: Optional pre-configured repository bundle.
                         If not provided, creates repositories based on session parameter.
            external_services: Optional pre-configured external services container.
                             If not provided, creates new container with settings.
            session: Optional database session for SQLAlchemy repositories.
                    If provided without repositories, creates SQLAlchemy-based bundle.
                    If neither provided, creates in-memory repositories.
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.session = session

        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        # Initialize or use provided repositories
        if repositories is None:
            self.repositories = self._create_repository_bundle(session)
        else:
            self.repositories = repositories

        # Initialize or use provided external services (includes ML services)
        if external_services is None:
            settings = get_settings()
            self.external_services = ExternalServicesContainer(settings)
        else:
            self.external_services = external_services

        # Initialize tasks with shared repositories and ML services
        self.infrastructure_task = RagInfrastructureTask(verbose=verbose)
        self.ingestion_task = RagIngestionTask(
            external_services=self.external_services,
            document_repository=self.repositories.document_repository,
            data_source_repository=self.repositories.data_source_repository,
            verbose=verbose,
        )
        self.embedding_task = RagEmbeddingTask(
            external_services=self.external_services,
            document_repository=self.repositories.document_repository,
            verbose=verbose,
        )

        # Summary tracking
        self.summary = RagBuildSummary()
        self.start_time: Optional[datetime] = None

    @staticmethod
    def _create_repository_bundle(
        session: Optional[AsyncSession] = None,
    ) -> RepositoryBundle:
        """Create a repository bundle based on session availability.

        Args:
            session: Optional database session

        Returns:
            RepositoryBundle with appropriate repository implementations
        """
        # Get settings
        settings = get_settings()

        # Create container with session (works for both database and in-memory)
        container = RepositoryContainer(settings, session)

        return RepositoryBundle(
            document_repository=container.document_repository,
            data_source_repository=container.data_source_repository,
        )

    async def run_infrastructure_check(
        self, spec: PipelineSpec
    ) -> RagInfrastructureTaskReport:
        """Run Task 1: Infrastructure validation independently.

        Args:
            spec: Pipeline specification

        Returns:
            RagInfrastructureTaskReport with task execution results and timing
        """
        logger.info("🔧 Task 1: Infrastructure Setup and Validation")

        try:
            # Phase now returns TaskReport directly
            task_report = await self.infrastructure_task.execute(spec)

            logger.info(
                f"✅ Task 1 completed in {task_report.execution_time_seconds:.2f}s"
            )
            return task_report

        except Exception as e:
            logger.error(f"❌ Task 1 failed: {str(e)}")
            raise

    async def run_ingestion(
        self, spec: PipelineSpec, dry_run: bool = False, enable_semantic_chunking: bool = False
    ) -> RagIngestionTaskReport:
        """Run Task 2: Document ingestion independently.

        Args:
            spec: Pipeline specification
            dry_run: Run in dry-run mode (skip actual storage)
            enable_semantic_chunking: If True, use semantic splitter with embeddings for chunk boundaries.
                                     If False, use sentence splitter (faster, no embeddings).
                                     Defaults to False for performance.

        Returns:
            RagIngestionTaskReport with task execution results and timing
        """
        logger.info("📄 Task 2: Document Ingestion and Processing")

        try:
            # Override storage for dry run
            if dry_run:
                data = spec.model_dump()
                data.setdefault("storage", {})
                data["storage"]["bucket"] = "rag-dry-run"
                data["storage"]["provider"] = "local"
                spec = PipelineSpec.model_validate(data)

            # Task now returns TaskReport directly
            task_report = await self.ingestion_task.execute(
                spec, dry_run=dry_run, enable_semantic_chunking=enable_semantic_chunking
            )

            documents_processed = (
                task_report.phase_result.processed
                if task_report.phase_result
                else 0
            )
            logger.info(
                f"✅ Task 2 completed in {task_report.execution_time_seconds:.2f}s: {documents_processed} documents processed"
            )
            return task_report

        except Exception as e:
            logger.error(f"❌ Task 2 failed: {str(e)}")
            raise

    async def run_embedding(
        self,
        spec: PipelineSpec,
        batch_size: int = 32,
        document_limit: Optional[int] = None,
        chunk_load_policy: ChunkLoadPolicy = ChunkLoadPolicy.EXISTING_ONLY,
    ) -> RagEmbeddingTaskReport:
        """Run Task 3: Embedding generation independently.

        This task loads existing chunks from Task 2 (Ingestion) and generates embeddings.
        It does NOT perform any chunking operations under normal circumstances.

        Args:
            spec: Pipeline specification
            batch_size: Batch size for embedding generation
            document_limit: Optional limit on documents to process
            chunk_load_policy: Policy for loading chunks from Task 2.
                             - EXISTING_ONLY (default): Strict mode - fail if chunks missing
                             - SENTENCE_FALLBACK: Use sentence splitter as fallback
                             - SEMANTIC_FALLBACK: Use semantic splitter as fallback
                             Default is EXISTING_ONLY for production-safe fail-fast behavior.

        Returns:
            RagEmbeddingTaskReport with task execution results and timing
        """
        logger.info("🔢 Task 3: Embedding Generation and Storage")

        try:
            # Task now returns TaskReport directly
            task_report = await self.embedding_task.execute(
                spec=spec,
                batch_size=batch_size,
                deduplication_enabled=True,
                max_retries=3,
                show_progress=not self.verbose,
                document_limit=document_limit,
                chunk_load_policy=chunk_load_policy,
            )

            embeddings_stored = (
                task_report.phase_result.embeddings_stored
                if task_report.phase_result
                else 0
            )
            logger.info(
                f"✅ Task 3 completed in {task_report.execution_time_seconds:.2f}s: {embeddings_stored} embeddings stored"
            )
            return task_report

        except Exception as e:
            logger.error(f"❌ Task 3 failed: {str(e)}")
            raise

    async def run_complete_pipeline(
        self,
        spec: PipelineSpec,
        dry_run: bool = False,
        batch_size: int = 32,
        document_limit: Optional[int] = None,
        output_dir: Optional[str] = None,
        enable_semantic_chunking: bool = False,
    ) -> RagBuildSummary:
        """Run the complete RAG pipeline (all 3 tasks sequentially).

        Args:
            spec: Pipeline specification
            dry_run: Run in dry-run mode (skip actual storage)
            batch_size: Batch size for embedding generation
            document_limit: Optional limit on documents to process
            output_dir: Directory to save reports and outputs
            enable_semantic_chunking: If True, Task 2 uses semantic splitter with embeddings.
                                     If False, uses sentence splitter (faster, no embeddings).
                                     Defaults to False for performance.

        Returns:
            RagBuildSummary with comprehensive results from all tasks
        """
        self.start_time = datetime.now(timezone.utc)
        self.summary.mark_started()

        logger.info("🚀 Starting RAG Build Pipeline")
        logger.info(f"Configuration: {spec}")
        logger.info(f"Output Directory: {output_dir or 'default'}")
        logger.info(f"Dry Run: {dry_run}")
        logger.info(f"Semantic Chunking (Task 2): {enable_semantic_chunking}")

        # Create output directory if specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        try:
            # Task 1: Infrastructure Validation
            await self._run_infrastructure_with_tracking(spec, output_dir)

            # Task 2: Document Ingestion (performs all chunking)
            await self._run_ingestion_with_tracking(spec, dry_run, output_dir, enable_semantic_chunking)

            # Task 3: Embedding Generation (loads chunks from Task 2, uses same chunking method for fallback)
            await self._run_embedding_with_tracking(
                spec, batch_size, document_limit, output_dir, enable_semantic_chunking
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
        """Run infrastructure task with result tracking."""
        try:
            # Phase now returns TaskReport directly with all timing
            task_report = await self.infrastructure_task.execute(spec)

            # Extract validation result from task report
            validation_result = task_report.phase_result
            if not validation_result:
                raise ValueError("Infrastructure phase returned no result")

            # Create phase result (for backward compatibility with RagBuildSummary)
            phase_result = create_infrastructure_phase_result(
                success=task_report.success,
                execution_time=task_report.execution_time_seconds,
                validation_results=validation_result.validation_results,
                errors=validation_result.errors,
                warnings=validation_result.warnings,
                start_time=task_report.start_time,
                end_time=task_report.end_time,
            )

            # Add to summary tracking
            self.summary.add_phase_result(phase_result)
            self.summary.infrastructure_task_report = task_report

            # Save phase-specific report
            if output_dir:
                self._save_phase_report(
                    output_dir, "task_1_infrastructure.json", task_report.as_dict()
                )

            logger.info(
                f"✅ Task 1 completed in {task_report.execution_time_seconds:.2f}s"
            )

        except Exception as e:
            logger.error(f"❌ Task 1 failed: {str(e)}")
            raise

    async def _run_ingestion_with_tracking(
        self, spec: PipelineSpec, dry_run: bool, output_dir: Optional[str], enable_semantic_chunking: bool
    ) -> None:
        """Run ingestion task with result tracking.

        Args:
            enable_semantic_chunking: If True, use semantic splitter with embeddings.
                                     If False, use sentence splitter (faster, no embeddings).
        """
        try:
            # Task now returns TaskReport directly with all timing
            task_report = await self.ingestion_task.execute(
                spec, dry_run=dry_run, enable_semantic_chunking=enable_semantic_chunking
            )

            # Extract ingestion summary from task report
            ingestion_summary = task_report.phase_result
            if not ingestion_summary:
                raise ValueError("Ingestion phase returned no result")

            # Create phase result (for backward compatibility with RagBuildSummary)
            phase_result = create_ingestion_phase_result(
                success=task_report.success,
                execution_time=task_report.execution_time_seconds,
                pipeline_summary=ingestion_summary.as_dict(),
                errors=[
                    str(error.get("message", error))
                    for error in ingestion_summary.errors
                ],
                start_time=task_report.start_time,
                end_time=task_report.end_time,
            )

            # Add to summary tracking
            self.summary.add_phase_result(phase_result)
            self.summary.ingestion_task_report = task_report

            # Save phase-specific reports
            if output_dir:
                self._save_phase_report(
                    output_dir, "task_2_ingestion.json", task_report.as_dict()
                )
                self._save_phase_report(
                    output_dir,
                    "ingestion_detailed.json",
                    ingestion_summary.as_dict(),
                )

            if ingestion_summary.processed > 0:
                logger.info(
                    f"✅ Task 2 completed in {task_report.execution_time_seconds:.2f}s: {ingestion_summary.processed} documents processed"
                )
            else:
                logger.warning("⚠️ Task 2 completed but no documents were processed")

        except Exception as e:
            logger.error(f"❌ Task 2 failed: {str(e)}")
            raise

    async def _run_embedding_with_tracking(
        self,
        spec: PipelineSpec,
        batch_size: int,
        document_limit: Optional[int],
        output_dir: Optional[str],
        enable_semantic_chunking: bool,
    ) -> None:
        """Run embedding task with result tracking.

        This task loads existing chunks from Task 2 (Ingestion) and generates embeddings.

        Args:
            enable_semantic_chunking: Used to determine chunk_load_policy.
                                     Maps to SEMANTIC_FALLBACK if True, SENTENCE_FALLBACK if False.
                                     This ensures fallback (if triggered) matches Task 2's chunking method.
        """
        try:
            # Map enable_semantic_chunking to appropriate chunk load policy
            # This preserves backward compatibility while using the new enum
            chunk_load_policy = (
                ChunkLoadPolicy.SEMANTIC_FALLBACK
                if enable_semantic_chunking
                else ChunkLoadPolicy.SENTENCE_FALLBACK
            )

            # Task now returns TaskReport directly with all timing
            task_report = await self.embedding_task.execute(
                spec=spec,
                batch_size=batch_size,
                deduplication_enabled=True,
                max_retries=3,
                show_progress=not self.verbose,
                document_limit=document_limit,
                chunk_load_policy=chunk_load_policy,
            )

            # Extract embedding result from task report
            embedding_result = task_report.phase_result
            if not embedding_result:
                raise ValueError("Embedding phase returned no result")

            # Create phase result (for backward compatibility with RagBuildSummary)
            phase_result = create_embedding_phase_result(
                success=task_report.success,
                execution_time=task_report.execution_time_seconds,
                embedding_result=embedding_result.as_dict(),
                errors=embedding_result.errors,
                start_time=task_report.start_time,
                end_time=task_report.end_time,
            )

            # Add to summary tracking
            self.summary.add_phase_result(phase_result)
            self.summary.embedding_task_report = task_report

            # Save phase-specific reports
            if output_dir:
                self._save_phase_report(
                    output_dir, "task_3_embedding.json", task_report.as_dict()
                )
                self._save_phase_report(
                    output_dir, "embedding_detailed.json", embedding_result.as_dict()
                )

            if embedding_result.success:
                logger.info(
                    f"✅ Task 3 completed in {task_report.execution_time_seconds:.2f}s: {embedding_result.embeddings_stored} embeddings stored"
                )
            else:
                logger.error("❌ Task 3 failed")
                raise Exception(
                    f"Embedding pipeline failed with {len(embedding_result.errors)} errors"
                )

        except Exception as e:
            logger.error(f"❌ Task 3 failed: {str(e)}")
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
        main_report_path = output_path / "rag_build_summary.json"
        self.summary.save_to_file(str(main_report_path))

        # Save human-readable report
        readable_report_path = output_path / "rag_build_report.txt"
        with open(readable_report_path, "w") as f:
            f.write(f"RAG Build Pipeline Report\n")
            f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            f.write(self.summary.as_json(indent=2))

        logger.info(f"📊 Reports saved to {output_dir}")
        logger.info(f"   - Main summary: {main_report_path}")
        logger.info(f"   - Detailed report: {readable_report_path}")
