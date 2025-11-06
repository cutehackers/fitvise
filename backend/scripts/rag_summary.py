#!/usr/bin/env python3
"""
RAG Build Summary Report Structure

Provides data structures and utilities for generating comprehensive reports
from the RAG build pipeline execution.

This module defines the RagBuildSummary class that aggregates results from
all three tasks of the RAG build pipeline.

Note: Task-specific TaskReport classes are now defined in their respective task files:
- RagInfrastructureTaskReport in infrastructure_task.py
- RagIngestionTaskReport in ingestion_task.py
- RagEmbeddingTaskReport in embedding_task.py
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class PhaseResult:
    """Results from a single phase of the RAG build pipeline."""

    phase_name: str
    success: bool
    execution_time_seconds: float
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "phase_name": self.phase_name,
            "success": self.success,
            "execution_time_seconds": round(self.execution_time_seconds, 2),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "results": self.results,
            "errors": self.errors,
            "warnings": self.warnings,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings)
        }


@dataclass
class RagBuildSummary:
    """Comprehensive summary of RAG build pipeline execution.

    Aggregates results from all three tasks:
    - Task 1: Infrastructure Setup and Validation
    - Task 2: Document Ingestion and Processing
    - Task 3: Embedding Generation and Storage
    """

    pipeline_name: str = "RAG Build Pipeline"
    version: str = "1.0"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_execution_time_seconds: float = 0.0

    # Phase Results (kept for backward compatibility with internal tracking)
    phase_results: List[PhaseResult] = field(default_factory=list)

    # Task Reports - These wrap the actual phase result models
    # (Actual TaskReport classes are defined in their respective phase files)
    infrastructure_task_report: Optional[Any] = None  # RagInfrastructureTaskReport from infrastructure_phase.py
    ingestion_task_report: Optional[Any] = None  # RagIngestionTaskReport from ingestion_phase.py
    embedding_task_report: Optional[Any] = None  # RagEmbeddingTaskReport from embedding_phase.py

    # Aggregated Metrics
    total_documents_processed: int = 0
    total_chunks_generated: int = 0
    total_embeddings_stored: int = 0
    total_errors: int = 0
    total_warnings: int = 0

    # Status
    success: bool = False
    failure_phase: Optional[str] = None

    def add_phase_result(self, phase_result: PhaseResult) -> None:
        """Add a phase result to the summary."""
        self.phase_results.append(phase_result)

        # Update end time and total execution time
        self.end_time = phase_result.end_time
        if self.start_time:
            self.total_execution_time_seconds = (self.end_time - self.start_time).total_seconds()

        # Update aggregated metrics
        self.total_errors += len(phase_result.errors)
        self.total_warnings += len(phase_result.warnings)

        # Update success status
        if not phase_result.success and self.success:
            self.success = False
            self.failure_phase = phase_result.phase_name


    def mark_started(self) -> None:
        """Mark the pipeline as started."""
        self.start_time = datetime.now(timezone.utc)
        self.success = True  # Assume success until a phase fails

    def mark_completed(self) -> None:
        """Mark the pipeline as completed."""
        self.end_time = datetime.now(timezone.utc)
        if self.start_time:
            self.total_execution_time_seconds = (self.end_time - self.start_time).total_seconds()

    def get_phase_result(self, phase_name: str) -> Optional[PhaseResult]:
        """Get result for a specific phase."""
        for result in self.phase_results:
            if result.phase_name == phase_name:
                return result
        return None

    def as_dict(self) -> Dict[str, Any]:
        """Convert the entire summary to a dictionary."""
        return {
            "pipeline_name": self.pipeline_name,
            "version": self.version,
            "execution": {
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "total_execution_time_seconds": round(self.total_execution_time_seconds, 2),
                "total_execution_time_formatted": self._format_duration(self.total_execution_time_seconds)
            },
            "status": {
                "success": self.success,
                "failure_phase": self.failure_phase,
                "phases_completed": len(self.phase_results),
                "total_errors": self.total_errors,
                "total_warnings": self.total_warnings
            },
            "task_reports": {
                "infrastructure": self.infrastructure_task_report.as_dict() if self.infrastructure_task_report else None,
                "ingestion": self.ingestion_task_report.as_dict() if self.ingestion_task_report else None,
                "embedding": self.embedding_task_report.as_dict() if self.embedding_task_report else None,
            },
            "phase_results": [phase.as_dict() for phase in self.phase_results],
            "aggregated_metrics": {
                "total_documents_processed": self.total_documents_processed,
                "total_chunks_generated": self.total_chunks_generated,
                "total_embeddings_stored": self.total_embeddings_stored,
                "average_chunks_per_document": round((self.total_chunks_generated / self.total_documents_processed), 2) if self.total_documents_processed > 0 else 0,
                "embedding_success_rate": round((self.total_embeddings_stored / self.total_chunks_generated * 100), 2) if self.total_chunks_generated > 0 else 0
            }
        }

    def as_json(self, indent: int = 2) -> str:
        """Convert the summary to JSON string."""
        return json.dumps(self.as_dict(), indent=indent, default=str)

    def save_to_file(self, file_path: str) -> None:
        """Save the summary to a JSON file."""
        with open(file_path, 'w') as f:
            f.write(self.as_json())

    def print_summary(self) -> None:
        """Print a formatted summary to the console."""
        print("\n" + "="*80)
        print(f"🚀 {self.pipeline_name} - EXECUTION SUMMARY")
        print("="*80)

        # Overall Status
        status_icon = "✅" if self.success else "❌"
        print(f"Status: {status_icon} {'SUCCESS' if self.success else 'FAILED'}")
        if self.failure_phase:
            print(f"Failure Phase: {self.failure_phase}")

        print(f"Execution Time: {self._format_duration(self.total_execution_time_seconds)}")
        print(f"Phases Completed: {len(self.phase_results)}/3")
        print(f"Total Errors: {self.total_errors}")
        print(f"Total Warnings: {self.total_warnings}")

        # Aggregated Metrics
        print("\n📊 AGGREGATED METRICS:")
        print(f"Documents Processed: {self.total_documents_processed}")
        print(f"Chunks Generated: {self.total_chunks_generated}")
        print(f"Embeddings Stored: {self.total_embeddings_stored}")

        if self.total_documents_processed > 0:
            avg_chunks = round((self.total_chunks_generated / self.total_documents_processed), 2)
            print(f"Average Chunks per Document: {avg_chunks}")

        if self.total_chunks_generated > 0:
            embedding_rate = round((self.total_embeddings_stored / self.total_chunks_generated * 100), 2)
            print(f"Embedding Success Rate: {embedding_rate}%")

        # Phase Results
        print("\n📋 PHASE RESULTS:")
        for phase in self.phase_results:
            phase_icon = "✅" if phase.success else "❌"
            print(f"  {phase_icon} Phase {phase.phase_name}: {round(phase.execution_time_seconds, 2)}s")

            if phase.errors:
                for error in phase.errors[:2]:  # Show first 2 errors
                    print(f"    ❌ {error}")
                if len(phase.errors) > 2:
                    print(f"    ... and {len(phase.errors) - 2} more errors")

        if self.total_warnings > 0:
            print(f"\n⚠️  {self.total_warnings} warnings across all phases")

        print("="*80)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in seconds to human readable format."""
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.2f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h {minutes}m {secs:.2f}s"


def create_infrastructure_phase_result(
    success: bool,
    execution_time: float,
    validation_results: Dict[str, Any],
    errors: List[str],
    warnings: List[str],
    start_time: datetime,
    end_time: datetime
) -> PhaseResult:
    """Create a PhaseResult for infrastructure validation."""
    return PhaseResult(
        phase_name="Infrastructure Setup",
        success=success,
        execution_time_seconds=execution_time,
        results=validation_results,
        errors=errors,
        warnings=warnings,
        start_time=start_time,
        end_time=end_time
    )


def create_ingestion_phase_result(
    success: bool,
    execution_time: float,
    pipeline_summary: Dict[str, Any],
    errors: List[str],
    start_time: datetime,
    end_time: datetime
) -> PhaseResult:
    """Create a PhaseResult for document ingestion."""

    # Extract ingestion-specific metrics
    counters = pipeline_summary.get("counters", {})

    return PhaseResult(
        phase_name="Document Ingestion",
        success=success,
        execution_time_seconds=execution_time,
        results={
            "discovered": counters.get("discovered", 0),
            "processed": pipeline_summary.get("processed", 0),
            "skipped": pipeline_summary.get("skipped", 0),
            "failed": pipeline_summary.get("failed", 0),
            "chunks_generated": counters.get("chunking", {}).get("total_chunks", 0),
            "stored_objects": len(pipeline_summary.get("stored", [])),
            "by_origin": counters.get("by_origin", {}),
            "total_errors": counters.get("total_errors", 0)
        },
        errors=errors,
        warnings=[],
        start_time=start_time,
        end_time=end_time
    )


def create_embedding_phase_result(
    success: bool,
    execution_time: float,
    embedding_result: Dict[str, Any],
    errors: List[str],
    start_time: datetime,
    end_time: datetime
) -> PhaseResult:
    """Create a PhaseResult for embedding generation."""

    return PhaseResult(
        phase_name="Embedding Generation",
        success=success,
        execution_time_seconds=execution_time,
        results={
            "documents_processed": embedding_result.get("documents_processed", 0),
            "total_chunks": embedding_result.get("total_chunks", 0),
            "unique_chunks": embedding_result.get("unique_chunks", 0),
            "duplicates_removed": embedding_result.get("duplicates_removed", 0),
            "embeddings_generated": embedding_result.get("embeddings_generated", 0),
            "embeddings_stored": embedding_result.get("embeddings_stored", 0),
            "processing_stats": embedding_result.get("processing_stats", {}),
            "deduplication_stats": embedding_result.get("deduplication_stats", {})
        },
        errors=errors,
        warnings=embedding_result.get("warnings", []),
        start_time=start_time,
        end_time=end_time
    )