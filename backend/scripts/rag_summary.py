#!/usr/bin/env python3
"""
RAG Ingestion Summary Report Structure

Provides data structures and utilities for generating comprehensive reports
from the RAG build pipeline execution.

This module defines the RagIngestionSummary class and related components
that aggregate results from all three phases of the RAG build pipeline.
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
class InfrastructureResults:
    """Specific results from infrastructure validation phase."""

    embedding_service_status: Dict[str, Any]
    weaviate_status: Dict[str, Any]
    object_storage_status: Dict[str, Any]
    configuration_status: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "embedding_service": self.embedding_service_status,
            "weaviate": self.weaviate_status,
            "object_storage": self.object_storage_status,
            "configuration": self.configuration_status
        }


@dataclass
class IngestionResults:
    """Specific results from document ingestion phase."""

    documents_discovered: int
    documents_processed: int
    documents_skipped: int
    documents_failed: int
    chunks_generated: int
    storage_objects_created: int
    processing_errors: List[str]

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "documents_discovered": self.documents_discovered,
            "documents_processed": self.documents_processed,
            "documents_skipped": self.documents_skipped,
            "documents_failed": self.documents_failed,
            "chunks_generated": self.chunks_generated,
            "storage_objects_created": self.storage_objects_created,
            "processing_errors": self.processing_errors,
            "processing_success_rate": round((self.documents_processed / self.documents_discovered * 100), 2) if self.documents_discovered > 0 else 0,
            "average_chunks_per_document": round((self.chunks_generated / self.documents_processed), 2) if self.documents_processed > 0 else 0
        }


@dataclass
class EmbeddingResults:
    """Specific results from embedding generation phase."""

    documents_processed: int
    total_chunks: int
    unique_chunks: int
    duplicates_removed: int
    embeddings_generated: int
    embeddings_stored: int
    deduplication_stats: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "documents_processed": self.documents_processed,
            "total_chunks": self.total_chunks,
            "unique_chunks": self.unique_chunks,
            "duplicates_removed": self.duplicates_removed,
            "duplicates_percentage": round((self.duplicates_removed / self.total_chunks * 100), 2) if self.total_chunks > 0 else 0,
            "embeddings_generated": self.embeddings_generated,
            "embeddings_stored": self.embeddings_stored,
            "embedding_success_rate": round((self.embeddings_stored / self.unique_chunks * 100), 2) if self.unique_chunks > 0 else 0,
            "deduplication_stats": self.deduplication_stats,
            "average_chunks_per_document": round((self.total_chunks / self.documents_processed), 2) if self.documents_processed > 0 else 0
        }


@dataclass
class RagIngestionSummary:
    """Comprehensive summary of RAG ingestion pipeline execution.

    Aggregates results from all three phases:
    - Phase 1: Infrastructure Setup and Validation
    - Phase 2: Document Ingestion and Processing
    - Phase 3: Embedding Generation and Storage
    """

    pipeline_name: str = "RAG Ingestion Pipeline"
    version: str = "1.0"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_execution_time_seconds: float = 0.0

    # Phase Results
    phase_results: List[PhaseResult] = field(default_factory=list)

    # Specific Results by Phase
    infrastructure_results: Optional[InfrastructureResults] = None
    ingestion_results: Optional[IngestionResults] = None
    embedding_results: Optional[EmbeddingResults] = None

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

    def set_infrastructure_results(self, results: InfrastructureResults) -> None:
        """Set infrastructure-specific results."""
        self.infrastructure_results = results

    def set_ingestion_results(self, results: IngestionResults) -> None:
        """Set ingestion-specific results."""
        self.ingestion_results = results
        self.total_documents_processed = results.documents_processed
        self.total_chunks_generated = results.chunks_generated

    def set_embedding_results(self, results: EmbeddingResults) -> None:
        """Set embedding-specific results."""
        self.embedding_results = results
        self.total_embeddings_stored = results.embeddings_stored

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
            "phase_results": [phase.as_dict() for phase in self.phase_results],
            "infrastructure_results": self.infrastructure_results.as_dict() if self.infrastructure_results else None,
            "ingestion_results": self.ingestion_results.as_dict() if self.ingestion_results else None,
            "embedding_results": self.embedding_results.as_dict() if self.embedding_results else None,
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