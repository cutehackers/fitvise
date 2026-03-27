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
from app.config.ml_models.chunking_configs import get_chunking_config
from app.pipeline.phases.rag_infrastructure_task import (
    InfrastructureResult,
    RagInfrastructureTaskReport,
)
from app.pipeline.phases.rag_embedding_task import (
    EmbeddingResult,
    RagEmbeddingTaskReport,
)
from app.pipeline.phases.rag_ingestion_task import RagIngestionTaskReport
from app.pipeline.contracts import RunSummary
from app.domain.entities.chunk_load_policy import ChunkLoadPolicy
from app.domain.repositories.document_repository import DocumentRepository
from app.domain.entities.chunk_load_policy import ChunkLoadPolicy
from app.domain.repositories.document_repository import DocumentRepository
from app.domain.repositories.data_source_repository import DataSourceRepository
from app.domain.repositories.embedding_repository import EmbeddingRepository

from app.di.containers.container import AppContainer
from app.di.containers.infra_container import InfraContainer
from app.core.settings import Settings, get_settings
from scripts.rag_summary import (
    RagBuildSummary,
    create_infrastructure_phase_result,
    create_ingestion_phase_result,
    create_embedding_phase_result,
)
from app.pipeline.chunking_config_resolver import resolve_chunking_configuration

logger = logging.getLogger(__name__)


@dataclass
class RepositoryBundle:
    """Bundle of shared repository instances for pipeline execution.

    This ensures all phases work with the same repository instances,
    maintaining data continuity throughout the pipeline.
    """

    document_repository: DocumentRepository
    document_repository: DocumentRepository
    data_source_repository: DataSourceRepository
    embedding_repository: EmbeddingRepository


class RAGWorkflow:
    """Orchestrates the complete RAG build pipeline with dependency injection.

    This class coordinates all three tasks of the RAG pipeline:
    1. Infrastructure validation
    2. Document ingestion and processing
    3. Embedding generation and storage

    All tasks share the same repository instances to ensure data continuity.

    ## Pipeline Configuration Options

    The RAGWorkflow uses chunking presets and overrides defined in the PipelineSpec to control
    the chunking strategy during document ingestion. The chunking behavior is determined by:
    1. Base preset (spec.chunking.preset)
    2. YAML override for semantic chunking (spec.chunking.enable_semantic_chunking)
    3. Additional parameter overrides (spec.chunking.overrides)

    ### Chunking Configuration (Task 2 - Document Ingestion)

    **Purpose**: Controls the chunking strategy used during document ingestion through YAML presets and overrides.

    **Configuration Options**:
    ```yaml
    chunking:
      enabled: true                              # Enable/disable chunking
      preset: balanced                           # Base preset: balanced, short_form, long_form, hierarchical
      enable_semantic_chunking: null             # Override: true=semantic, false=sentence, null=use preset default
      overrides: {}                              # Additional parameter overrides
    ```

    **Available Presets**:

    **balanced** (Default - Quality Mode):
    - Uses semantic chunking with embedding analysis
    - Generates temporary embeddings to determine semantic boundaries
    - Chunks are created where topic shifts occur (based on embedding similarity)
    - Higher computational cost but better semantic coherence
    - Optimal for: High-quality retrieval, document sets where context matters,
                  academic/technical content requiring coherent chunks

    **short_form** (Performance Mode):
    - Uses sentence-based splitting with fixed size boundaries
    - No embedding generation required during chunking
    - Faster processing, lower computational cost
    - Chunks are created based on sentence boundaries and size limits
    - Optimal for: Large document sets, processing speed prioritized, initial indexing

    **long_form**:
    - Semantic chunking optimized for long documents
    - Larger chunk sizes for better context preservation
    - Optimal for: Manuals, research papers, technical documentation

    **hierarchical**:
    - Multi-level chunking preserving document structure
    - Creates nested chunks at different granularity levels
    - Optimal for: Complex documents with clear hierarchical structure

    **Performance Considerations**:
    - short_form: 2-3x faster processing, minimal resource usage
    - balanced/long_form/hierarchical: Slower processing due to embedding generation

    ### Task Separation (Chunking in Task 2, Embedding in Task 3)

    **Architecture**: Clear separation of concerns between pipeline tasks.

    **Task 2 (Ingestion)**: Responsible for all chunking operations
    - Processes documents and generates chunks
    - Stores chunks in repository for Task 3 to consume
    - Uses chunking strategy defined in spec.chunking.preset

    **Task 3 (Embedding Generation)**: Consumes chunks from Task 2
    - Loads existing chunks from repository (created in Task 2)
    - Validates chunk availability before proceeding
    - Generates and stores embeddings for existing chunks only
    - Does NOT perform any chunking operations

    **Important**: If you need to change chunking strategy, re-run Task 2 (Ingestion) with
    a different preset. Task 3 will always use chunks created by Task 2.

    ### Recommended Configurations

    **Production Pipeline (Fast - short_form preset)**:
    ```yaml
    chunking:
      preset: short_form
      enable_semantic_chunking: false  # Explicit sentence-based chunking
    ```

    **High-Quality Pipeline (balanced preset with override)**:
    ```yaml
    chunking:
      preset: short_form     # Start with fast preset
      enable_semantic_chunking: true  # Override to enable semantic chunking
    ```

    **Hybrid Pipeline (balanced preset with sentence override)**:
    ```yaml
    chunking:
      preset: balanced
      enable_semantic_chunking: false  # Override balanced to use sentence splitting
    ```

    **Pipeline Experimentation**:
    ```python
    # Run ingestion with different configurations
    spec_fast = PipelineSpec(
        documents=DocumentOption(path="./data"),
        chunking=ChunkingOptions(preset="short_form", enable_semantic_chunking=False)
    )
    await workflow.run_ingestion(spec_fast)

    spec_quality = PipelineSpec(
        documents=DocumentOption(path="./data"),
        chunking=ChunkingOptions(preset="balanced", enable_semantic_chunking=True)
    )
    await workflow.run_ingestion(spec_quality)

    spec_hybrid = PipelineSpec(
        documents=DocumentOption(path="./data"),
        chunking=ChunkingOptions(preset="balanced", enable_semantic_chunking=False)
    )
    await workflow.run_ingestion(spec_hybrid)
    ```

    **Pipeline Debugging**:
    ```python
    # Test with small document limit and explicit configuration
    spec_debug = PipelineSpec(
        documents=DocumentOption(path="./data"),
        chunking=ChunkingOptions(preset="short_form", enable_semantic_chunking=False),
        limits=LimitOptions(max_files=10)
    )
    await workflow.run_ingestion(spec_debug)
    ```

    The chunking presets provide fine-grained control over the quality-performance tradeoff
    and enable flexible pipeline execution patterns for different use cases.
    """

    def __init__(
        self,
        repositories: Optional[RepositoryBundle] = None,
        container: Optional[AppContainer] = None,
        session: Optional[AsyncSession] = None,
        verbose: bool = False,
    ):
        """Initialize the RAG workflow orchestrator.

        Args:
            repositories: Optional pre-configured repository bundle.
                         If not provided, creates repositories based on session parameter.
            container: Optional pre-configured AppContainer.
                      If not provided, creates new AppContainer with settings.
            session: Optional database session for SQLAlchemy repositories.
                    If provided without repositories, creates SQLAlchemy-based bundle.
                    If neither provided, creates in-memory repositories.
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.session = session

        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        # Initialize or use provided container
        if container is None:
            self.container = AppContainer()
        else:
            self.container = container

        # Extract infra container for easier access
        self.infra = self.container.infra

        # Initialize or use provided repositories
        if repositories is None:
            # Repository bundle will be created lazily when needed
            self.repositories = None
            self._session = session
        else:
            self.repositories = repositories

        # Initialize tasks (repositories will be created lazily)
        self.infrastructure_task = RagInfrastructureTask(
            verbose=verbose
        )
        self.ingestion_task = None
        self.embedding_task = None

        # Summary tracking
        self.summary = RagBuildSummary()
        self.start_time: Optional[datetime] = None

    @staticmethod
    async def _create_repository_bundle(
        session: Optional[AsyncSession] = None,
        infra: Optional[InfraContainer] = None,
    ) -> RepositoryBundle:
        """Create a repository bundle based on session availability.

        Args:
            session: Optional database session
            infra: Optional infra container, needed for EmbeddingRepository

        Returns:
            RepositoryBundle with appropriate repository implementations
        """
        if infra is None:
            raise ValueError("InfraContainer is required for repository bundle creation")

        # Use repositories directly from InfraContainer
        # Note: Some providers may return coroutines, so we await them if needed
        doc_repo = infra.document_repository()
        if hasattr(doc_repo, '__await__'):
            doc_repo = await doc_repo

        data_repo = infra.data_source_repository()
        if hasattr(data_repo, '__await__'):
            data_repo = await data_repo

        embed_repo = infra.embedding_repository()
        if hasattr(embed_repo, '__await__'):
            embed_repo = await embed_repo

        return RepositoryBundle(
            document_repository=doc_repo,
            data_source_repository=data_repo,
            embedding_repository=embed_repo,
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
        self, spec: PipelineSpec, dry_run: bool = False
    ) -> RagIngestionTaskReport:
        """Run Task 2: Document ingestion independently.

        Args:
            spec: Pipeline specification
            dry_run: Run in dry-run mode (skip actual storage)

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

            # Create repository bundle if not exists
            if self.repositories is None:
                self.repositories = await self._create_repository_bundle(
                    self._session, self.infra
                )

            # Initialize ingestion task if not exists
            if self.ingestion_task is None:
                self.ingestion_task = RagIngestionTask(
                    infra=self.infra,
                    document_repository=self.repositories.document_repository,
                    data_source_repository=self.repositories.data_source_repository,
                    verbose=self.verbose,
                )

            # Task now returns TaskReport directly
            task_report = await self.ingestion_task.execute(
                spec, dry_run=dry_run
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
            # Create repository bundle if not exists
            if self.repositories is None:
                self.repositories = await self._create_repository_bundle(
                    self._session, self.infra
                )

            # Initialize embedding task if not exists
            if self.embedding_task is None:
                self.embedding_task = RagEmbeddingTask(
                    infra=self.infra,
                    document_repository=self.repositories.document_repository,
                    embedding_repository=self.repositories.embedding_repository,
                    verbose=self.verbose,
                )

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
    ) -> RagBuildSummary:
        """Run the complete RAG pipeline (all 3 tasks sequentially).

        Args:
            spec: Pipeline specification
            dry_run: Run in dry-run mode (skip actual storage)
            batch_size: Batch size for embedding generation
            document_limit: Optional limit on documents to process
            output_dir: Directory to save reports and outputs

        Returns:
            RagBuildSummary with comprehensive results from all tasks
        """
        import uuid
        self.start_time = datetime.now(timezone.utc)
        self.summary.mark_started()

        # Generate unique workflow ID for end-to-end tracing
        workflow_id = str(uuid.uuid4())

        logger.info("🚀 Starting RAG Build Pipeline")
        logger.info(f"Configuration: {spec}")
        logger.info(f"Output Directory: {output_dir or 'default'}")
        logger.info(f"Dry Run: {dry_run}")
        logger.info(f"Workflow ID: {workflow_id}")
        # Get chunking preset for logging
        chunking_config = resolve_chunking_configuration(spec)
        logger.info(f"Chunking Configurations: {chunking_config}")
        logger.info(f"Semantic Chunking: {chunking_config.get('enable_semantic_chunking', True)}")

        # Create output directory if specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        try:
            # Task 1: Infrastructure Validation
            await self._run_infrastructure_with_tracking(spec, output_dir, workflow_id)

            # Task 2: Document Ingestion (performs all chunking)
            await self._run_ingestion_with_tracking(spec, dry_run, output_dir, workflow_id)

            # Task 3: Embedding Generation (loads chunks from Task 2, uses same chunking method for fallback)
            await self._run_embedding_with_tracking(
                spec, batch_size, document_limit, output_dir, workflow_id
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
        self, spec: PipelineSpec, output_dir: Optional[str], workflow_id: Optional[str] = None
    ) -> None:
        """Run infrastructure task with result tracking.

        Args:
            spec: Pipeline specification
            output_dir: Optional output directory for reports
            workflow_id: Optional workflow ID for tracing
        """
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
        self, spec: PipelineSpec, dry_run: bool, output_dir: Optional[str], workflow_id: Optional[str] = None
    ) -> None:
        """Run ingestion task with result tracking.

        Args:
            spec: Pipeline specification containing chunking configuration
            dry_run: Run in dry-run mode
            output_dir: Optional output directory for reports
            workflow_id: Optional workflow ID for tracing
        """
        try:
            # Create repository bundle if not exists
            if self.repositories is None:
                self.repositories = await self._create_repository_bundle(
                    self._session, self.infra
                )

            # Initialize ingestion task if not exists
            if self.ingestion_task is None:
                self.ingestion_task = RagIngestionTask(
                    infra=self.infra,
                    document_repository=self.repositories.document_repository,
                    data_source_repository=self.repositories.data_source_repository,
                    verbose=self.verbose,
                )

            # Task now returns TaskReport directly with all timing
            task_report = await self.ingestion_task.execute(
                spec, dry_run=dry_run
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
        workflow_id: Optional[str] = None,
    ) -> None:
        """Run embedding task with result tracking.

        This task loads existing chunks from Task 2 (Ingestion) and generates embeddings.

        Args:
            spec: Pipeline specification containing chunking configuration
            batch_size: Batch size for embedding generation
            document_limit: Optional limit on documents to process
            output_dir: Optional output directory for reports
            workflow_id: Optional workflow ID for tracing
        """
        try:
            # Create repository bundle if not exists
            if self.repositories is None:
                self.repositories = await self._create_repository_bundle(
                    self._session, self.infra
                )

            # Initialize embedding task if not exists
            if self.embedding_task is None:
                self.embedding_task = RagEmbeddingTask(
                    infra=self.infra,
                    document_repository=self.repositories.document_repository,
                    embedding_repository=self.repositories.embedding_repository,
                    verbose=self.verbose,
                )

            # Determine chunk load policy based on the preset used in Task 2
            # This ensures fallback (if triggered) matches Task 2's chunking method
            chunking_config = resolve_chunking_configuration(spec)
            enable_semantic = chunking_config.get('enable_semantic_chunking', True)
            chunk_load_policy = (
                ChunkLoadPolicy.SEMANTIC_FALLBACK
                if enable_semantic
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
