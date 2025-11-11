"""Chunking configuration resolver for RAG pipeline.

This module provides functions to resolve chunking configurations from presets
and apply user-specified overrides from YAML configuration.

The resolver handles the complex logic of merging preset configurations with
user-defined overrides, ensuring consistent behavior across the pipeline.
"""

from typing import Dict, Any, Optional

from app.pipeline.config import PipelineSpec
from app.config.ml_models import get_chunking_config


def resolve_chunking_configuration(spec: PipelineSpec) -> Dict[str, Any]:
    """Resolve chunking configuration by merging preset with YAML overrides.

    This function performs the configuration resolution process:
    1. Loads the base preset configuration (defaults to 'balanced')
    2. Applies user-specified YAML overrides
    3. Returns the final configuration ready for use

    Args:
        spec: Pipeline specification containing chunking preferences and overrides

    Returns:
        Resolved chunking configuration dictionary with all overrides applied

    Example:
        >>> spec = PipelineSpec.from_file("rag_pipeline.yaml")
        >>> config = resolve_chunking_configuration(spec)
        >>> # config now contains the final chunking settings
    """
    # Load base configuration from preset (fallback to 'balanced' if not specified)
    preset_name = spec.chunking.preset or "balanced"
    chunk_config = get_chunking_config(preset_name)

    # Apply user-specified enable_semantic override from YAML
    if spec.chunking.enable_semantic is not None:
        chunk_config["enable_semantic"] = spec.chunking.enable_semantic

    # Apply any additional configuration overrides from YAML
    overrides = getattr(spec.chunking, "overrides", {})
    if overrides:
        chunk_config.update(overrides)

    return chunk_config


def requires_embedding_model(spec: PipelineSpec) -> bool:
    """Check if the chunking configuration requires an embedding model.

    This function determines whether semantic chunking will be used,
    which requires loading an embedding model for computing semantic boundaries.

    Args:
        spec: Pipeline specification containing chunking configuration

    Returns:
        True if semantic chunking is enabled (requires embedding model),
        False if sentence-based chunking will be used (no embedding model needed)

    Example:
        >>> spec = PipelineSpec.from_file("rag_pipeline.yaml")
        >>> if requires_embedding_model(spec):
        ...     print("Initialize embedding model")
        ... else:
        ...     print("Skip embedding model initialization")
    """
    config = resolve_chunking_configuration(spec)
    return config.get("enable_semantic", True)