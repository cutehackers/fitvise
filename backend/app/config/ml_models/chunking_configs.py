"""Chunking configuration presets for Task 2.1.1."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class ChunkingConfigurations:
    """Configuration template for a chunking strategy."""

    name: str
    description: str
    chunk_size: int = 1024
    chunk_overlap: int = 128
    min_chunk_chars: int = 80
    max_chunk_chars: int = 2048
    metadata_passthrough: Iterable[str] = field(
        default_factory=lambda: ("document_id", "source_id", "file_name", "doc_type")
    )
    chunk_sizes: Iterable[int] | None = None  # For hierarchical chunking (Task 2.1.3)
    enable_semantic: bool = True  # Use semantic chunking with embeddings

    def as_dict(self) -> Dict[str, int | List[str] | List[int] | str | bool]:
        result = {
            "preset": self.name,
            "description": self.description,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "min_chunk_chars": self.min_chunk_chars,
            "max_chunk_chars": self.max_chunk_chars,
            "metadata_passthrough_fields": list(self.metadata_passthrough),
            "enable_semantic": self.enable_semantic,
        }
        if self.chunk_sizes is not None:
            result["chunk_sizes"] = list(self.chunk_sizes)
        return result


_PRESETS: Dict[str, ChunkingConfigurations] = {
    "balanced": ChunkingConfigurations(
        name="balanced",
        description="Balanced splitter for mixed knowledge base documents.",
        chunk_size=768,
        chunk_overlap=128,
        min_chunk_chars=512,
        max_chunk_chars=3072,
        enable_semantic=True,  # Semantic for mixed content types
    ),
    "short_form": ChunkingConfigurations(
        name="short_form",
        description="Shorter chunks for FAQ or policy snippets.",
        chunk_size=512,
        chunk_overlap=64,
        min_chunk_chars=60,
        max_chunk_chars=1200,
        enable_semantic=False,  # Fast sentence-based for short content
    ),
    "long_form": ChunkingConfigurations(
        name="long_form",
        description="Long-form content such as manuals or research papers.",
        chunk_size=1536,
        chunk_overlap=256,
        min_chunk_chars=160,
        max_chunk_chars=3072,
        enable_semantic=True,  # Semantic for better coherence in long content
    ),
    "hierarchical": ChunkingConfigurations(
        name="hierarchical",
        description="Recursive chunking preserving policy/section/paragraph hierarchy.",
        chunk_size=2048,  # Largest level (root/document context)
        chunk_overlap=200,
        min_chunk_chars=100,
        max_chunk_chars=2048,
        chunk_sizes=[2048, 512, 128],  # Multi-level: document → section → paragraph
        enable_semantic=True,  # Semantic for document structure understanding
    ),
}


def list_chunking_presets() -> List[str]:
    """Return available preset names."""
    return list(_PRESETS.keys())


def get_chunking_config(preset: str | None = None) -> Dict[str, int | List[str] | str]:
    """Fetch a preset configuration as a dictionary for the chunker use case."""
    if preset:
        preset_lower = preset.lower()
        if preset_lower not in _PRESETS:
            raise ValueError(f"Unknown chunking preset: {preset}")
        return _PRESETS[preset_lower].as_dict()
    return _PRESETS["balanced"].as_dict()
