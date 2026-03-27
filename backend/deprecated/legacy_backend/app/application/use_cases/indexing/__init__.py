"""Indexing use cases for Phase 2 RAG system.

This module contains use cases for vector database setup, schema design,
and embedding ingestion pipeline operations.
"""

from .build_ingestion_pipeline import (
    BuildIngestionPipelineRequest,
    BuildIngestionPipelineResponse,
    BuildIngestionPipelineUseCase,
    DeduplicationStats,
)

__all__ = [
    "BuildIngestionPipelineRequest",
    "BuildIngestionPipelineResponse",
    "BuildIngestionPipelineUseCase",
    "DeduplicationStats",
]