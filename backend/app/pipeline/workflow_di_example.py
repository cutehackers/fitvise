"""RAG Workflow with Dependency Injection.

This file demonstrates the updated pipeline workflow using the new DI system.
Compare this with the original workflow.py to see the transformation.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from app.di import container
from app.di.container import (
    PipelineWorkflowProvider,
    DocumentRepositoryProvider,
    EmbeddingRepositoryProvider,
    ExternalServicesContainerProvider,
)

logger = logging.getLogger(__name__)

# ============================================================================
# NEW: Dependency Injection - Clean, Testable, Maintainable
# ============================================================================

@dataclass
class DIWorkflowBundle:
    """Bundle of dependencies provided by DI container.
    
    This replaces the old manual RepositoryBundle with a DI-provided
    bundle that includes all necessary dependencies.
    """
    infrastructure_task: RagInfrastructureTask
    ingestion_task: RagIngestionTask
    embedding_task: RagEmbeddingTask
    settings: Settings
    
    def __post_init__(self):
        """Validate that all dependencies are properly provided."""
        if not all([self.infrastructure_task, self.ingestion_task, self.embedding_task]):
            raise ValueError("All workflow dependencies must be provided")


class DI_RAGWorkflow:
    """RAG Workflow using dependency injection.
    
    This class demonstrates the benefits of DI for complex workflows:
    1. No manual service creation
    2. Automatic dependency resolution
    3. Easy testing with mock dependencies
    4. Clear dependency requirements
    5. Proper lifecycle management
    """

    def __init__(
        self,
        # All dependencies injected automatically by DI container
        infrastructure_task: RagInfrastructureTask = Depends(
            container.services.infrastructure_task.provider
        ),
        ingestion_task: RagIngestionTask = Depends(
            container.services.ingestion_task.provider
        ),
        embedding_task: RagEmbeddingTask = Depends(
            container.services.embedding_task.provider
        ),
        settings: Settings = Depends(container.settings.provider),
        verbose: bool = False,
    ):
        """Initialize workflow with DI-managed dependencies.
        
        Args:
            infrastructure_task: DI-managed infrastructure task
            ingestion_task: DI-managed ingestion task
            embedding_task: DI-managed embedding task
            settings: DI-managed application settings
            verbose: Enable verbose logging
        """
        # Dependencies provided by DI container - no manual creation
        self.infrastructure_task = infrastructure_task
        self.ingestion_task = ingestion_task
        self.embedding_task = embedding_task
        self.settings = settings
        
        self.verbose = verbose
        self.start_time: Optional[datetime] = None

        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        logger.info("✅ DI_RAGWorkflow initialized with dependency injection")
        logger.info(f"🔧 Infrastructure task: {type(infrastructure_task).__name__}")
        logger.info(f"📄 Ingestion task: {type(ingestion_task).__name__}")
        logger.info(f"🔢 Embedding task: {type(embedding_task).__name__}")
        logger.info(f"⚙️ Environment: {settings.environment}")

    async def run_complete_pipeline(
        self,
        spec: PipelineSpec,
        dry_run: bool = False,
        batch_size: int = 32,
        document_limit: Optional[int] = None,
        output_dir: Optional[str] = None,
    ) -> RagBuildSummary:
        """Run complete RAG pipeline with DI-managed tasks.
        
        Benefits shown:
        1. Tasks are pre-configured by DI container
        2. No manual dependency wiring required
        3. Automatic resource management
        4. Easy testing with task mocks
        """
        import uuid
        self.start_time = datetime.now(timezone.utc)
        
        # Create summary for tracking
        summary = RagBuildSummary()
        summary.mark_started()
        
        # Generate workflow ID for tracing
        workflow_id = str(uuid.uuid4())
        
        logger.info("🚀 Starting RAG Build Pipeline with DI")
        logger.info(f"Workflow ID: {workflow_id}")
        logger.info(f"Configuration: {spec}")
        logger.info(f"Dry Run: {dry_run}")
        logger.info(f"DI System: Active")
        
        try:
            # Task 1: Infrastructure validation (DI-managed task)
            await self._run_infrastructure_with_di(spec, output_dir, workflow_id, summary)
            
            # Task 2: Document ingestion (DI-managed task)
            await self._run_ingestion_with_di(spec, dry_run, output_dir, workflow_id, summary)
            
            # Task 3: Embedding generation (DI-managed task)
            await self._run_embedding_with_di(
                spec, batch_size, document_limit, output_dir, workflow_id, summary
            )
            
            # Complete pipeline
            summary.mark_completed()
            
            if output_dir:
                await self._save_reports(output_dir, summary, workflow_id)
            
            logger.info("✅ RAG pipeline completed successfully with DI")
            return summary
            
        except Exception as e:
            logger.error(f"❌ RAG pipeline failed: {str(e)}")
            summary.success = False
            if not summary.failure_phase:
                summary.failure_phase = "Unknown"
            summary.mark_completed()
            raise

    async def _run_infrastructure_with_di(
        self, spec: PipelineSpec, output_dir: Optional[str], workflow_id: str, summary: RagBuildSummary
    ) -> None:
        """Run infrastructure task with DI-managed task instance.
        
        Benefits:
        1. Task is fully configured by DI container
        2. No manual task initialization required
        3. Automatic dependency injection within task
        4. Easy testing with task mocks
        """
        logger.info("🔧 Task 1: Infrastructure Setup (DI-managed)")
        
        try:
            # Task is ready to use - no initialization needed
            task_report = await self.infrastructure_task.execute(spec)
            
            logger.info(f"✅ Task 1 completed in {task_report.execution_time_seconds:.2f}s")
            
            # Add to summary
            summary.add_phase_result(task_report.phase_result)
            summary.infrastructure_task_report = task_report
            
        except Exception as e:
            logger.error(f"❌ Task 1 failed: {str(e)}")
            raise

    async def _run_ingestion_with_di(
        self, spec: PipelineSpec, dry_run: bool, output_dir: Optional[str], workflow_id: str, summary: RagBuildSummary
    ) -> None:
        """Run ingestion task with DI-managed task instance.
        
        Benefits:
        1. Task automatically has all required dependencies
        2. Repository instances provided by DI
        3. External services injected automatically
        4. Consistent configuration across tasks
        """
        logger.info("📄 Task 2: Document Ingestion (DI-managed)")
        
        try:
            # Task execution with DI-provided dependencies
            task_report = await self.ingestion_task.execute(spec, dry_run=dry_run)
            
            ingestion_summary = task_report.phase_result
            if ingestion_summary and ingestion_summary.processed > 0:
                logger.info(
                    f"✅ Task 2 completed in {task_report.execution_time_seconds:.2f}s: "
                    f"{ingestion_summary.processed} documents processed"
                )
            else:
                logger.warning("⚠️ Task 2 completed but no documents were processed")
            
            # Add to summary
            summary.add_phase_result(task_report.phase_result)
            summary.ingestion_task_report = task_report
            
        except Exception as e:
            logger.error(f"❌ Task 2 failed: {str(e)}")
            raise

    async def _run_embedding_with_di(
        self,
        spec: PipelineSpec,
        batch_size: int,
        document_limit: Optional[int],
        output_dir: Optional[str],
        workflow_id: str,
        summary: RagBuildSummary,
    ) -> None:
        """Run embedding task with DI-managed task instance.
        
        Benefits:
        1. Embedding service automatically initialized
        2. Weaviate client automatically connected
        3. Repository instances provided by DI
        4. Configuration consistency guaranteed
        """
        logger.info("🔢 Task 3: Embedding Generation (DI-managed)")
        
        try:
            # Determine chunk load policy based on configuration
            chunking_config = resolve_chunking_configuration(spec)
            enable_semantic = chunking_config.get('enable_semantic_chunking', True)
            chunk_load_policy = (
                ChunkLoadPolicy.SEMANTIC_FALLBACK
                if enable_semantic
                else ChunkLoadPolicy.SENTENCE_FALLBACK
            )
            
            # Task execution with all dependencies provided by DI
            task_report = await self.embedding_task.execute(
                spec=spec,
                batch_size=batch_size,
                deduplication_enabled=True,
                max_retries=3,
                show_progress=not self.verbose,
                document_limit=document_limit,
                chunk_load_policy=chunk_load_policy,
            )
            
            embedding_result = task_report.phase_result
            if embedding_result and embedding_result.success:
                logger.info(
                    f"✅ Task 3 completed in {task_report.execution_time_seconds:.2f}s: "
                    f"{embedding_result.embeddings_stored} embeddings stored"
                )
            else:
                logger.error("❌ Task 3 failed")
                raise Exception(
                    f"Embedding pipeline failed with {len(embedding_result.errors) if embedding_result else 0} errors"
                )
            
            # Add to summary
            summary.add_phase_result(task_report.phase_result)
            summary.embedding_task_report = task_report
            
        except Exception as e:
            logger.error(f"❌ Task 3 failed: {str(e)}")
            raise

    async def _save_reports(self, output_dir: str, summary: RagBuildSummary, workflow_id: str) -> None:
        """Save pipeline reports with DI context.
        
        Benefits:
        1. Settings provided by DI for report generation
        2. Consistent report formatting
        3. Easy testing of report generation
        """
        import json
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Main summary report
        main_report_path = output_path / "rag_build_summary.json"
        summary.save_to_file(str(main_report_path))
        
        # DI-enhanced report with dependency information
        di_report_path = output_path / "di_system_report.json"
        di_report = {
            "workflow_id": workflow_id,
            "completion_time": datetime.now(timezone.utc).isoformat(),
            "di_system": {
                "status": "active",
                "providers_used": [
                    "config.settings",
                    "external.sentence_transformer_service", 
                    "external.weaviate_client",
                    "repositories.document_repository",
                    "repositories.embedding_repository",
                    "services.infrastructure_task",
                    "services.ingestion_task", 
                    "services.embedding_task",
                ],
                "environment": self.settings.environment,
                "debug_mode": self.settings.debug,
            },
            "pipeline_summary": summary.as_dict(),
        }
        
        with open(di_report_path, "w") as f:
            json.dump(di_report, f, indent=2)
        
        logger.info(f"📊 Reports saved to {output_dir}")
        logger.info(f"   - Main summary: {main_report_path}")
        logger.info(f"   - DI system report: {di_report_path}")

# ============================================================================
# Usage Examples and Testing
# ============================================================================

"""
Example usage showing how simple workflow creation becomes with DI:

# Before: Manual workflow creation with complex dependency management
def create_legacy_workflow():
    settings = get_settings()
    
    # Manual external services container creation
    external_services = ExternalServicesContainer(settings)
    
    # Manual repository container creation with session
    async with AsyncSessionLocal() as session:
        repositories = RepositoryContainer(settings, session, external_services)
        
        # Manual task creation and wiring
        infrastructure_task = RagInfrastructureTask(verbose=True)
        ingestion_task = RagIngestionTask(
            external_services=external_services,
            document_repository=repositories.document_repository,
            data_source_repository=repositories.data_source_repository,
        )
        embedding_task = RagEmbeddingTask(
            external_services=external_services,
            document_repository=repositories.document_repository,
            embedding_repository=repositories.embedding_repository,
        )
        
        # Manual workflow creation
        workflow = RAGWorkflow(
            infrastructure_task=infrastructure_task,
            ingestion_task=ingestion_task,
            embedding_task=embedding_task,
        )
        
        return workflow

# After: Simple workflow creation with DI
def create_di_workflow():
    from app.di import container
    
    # Single call gets fully configured workflow
    workflow = container.services.pipeline_workflow()
    
    return workflow

# Usage
async def run_pipeline():
    # DI-managed workflow - no manual setup required
    workflow = create_di_workflow()
    
    spec = PipelineSpec(
        documents=DocumentOption(path="./data"),
        chunking=ChunkingOptions(preset="balanced"),
    )
    
    result = await workflow.run_complete_pipeline(spec)
    return result


Testing examples showing how easy testing becomes:

def test_workflow_with_mocks():
    from app.di.testing import create_test_container
    from app.di import container
    
    # Create test container with mocked dependencies
    test_container = create_test_container()
    
    with container.override(test_container):
        # Workflow automatically uses mocked dependencies
        workflow = container.services.pipeline_workflow()
        
        spec = PipelineSpec(
            documents=DocumentOption(path="./test_data"),
        )
        
        result = await workflow.run_complete_pipeline(spec)
        
        # Verify workflow used mocked dependencies
        assert result.success
        assert result.documents_processed >= 0  # Mocked data

def test_workflow_with_selective_mocks():
    from app.di.testing import create_test_container, TestOverrides
    from app.di import container
    
    test_container = create_test_container()
    
    # Mock only the embedding service, keep other services real
    TestOverrides.with_mock_embedding_service(
        test_container,
        mock_embedding=[0.5, 0.6, 0.7]
    )
    
    with container.override(test_container):
        workflow = container.services.pipeline_workflow()
        
        # Workflow will use real repositories but mocked embedding service
        spec = PipelineSpec(...)
        result = await workflow.run_complete_pipeline(spec)
        
        assert result.success

Benefits demonstrated:
1. Dramatically simplified workflow creation
2. Automatic dependency resolution
3. Easy selective mocking for testing
4. No manual service creation or wiring
5. Consistent configuration management
6. Clear dependency requirements
"""
