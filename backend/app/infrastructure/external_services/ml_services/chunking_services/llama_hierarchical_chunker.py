"""LlamaIndex hierarchical chunking service for recursive document splitting (Task 2.1.3)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import uuid4

from app.domain.exceptions import ChunkingDependencyError, ChunkGenerationError

logger = logging.getLogger(__name__)

try:  # Optional dependency - degrade gracefully if unavailable
    from llama_index.core import Document as _LlamaDocument  # type: ignore
    from llama_index.core.node_parser import (  # type: ignore
        HierarchicalNodeParser as _HierarchicalNodeParser,
    )
    from llama_index.core.node_parser import SentenceSplitter as _SentenceSplitter  # type: ignore
except Exception:  # pragma: no cover - llama_index not installed
    _LlamaDocument = None  # type: ignore
    _HierarchicalNodeParser = None  # type: ignore
    _SentenceSplitter = None  # type: ignore


class HierarchicalChunkingServiceError(ChunkGenerationError):
    """Raised when hierarchical chunking fails due to unexpected runtime errors."""


@dataclass
class HierarchicalChunkerConfig:
    """Runtime configuration for hierarchical chunking."""

    chunk_sizes: List[int] = field(default_factory=lambda: [2048, 512, 128])
    chunk_overlap: int = 200
    min_chunk_chars: int = 100
    max_chunk_chars: int = 2048
    preserve_hierarchy: bool = True
    metadata_passthrough_fields: Sequence[str] = ("document_id", "source_id", "file_name", "doc_type")
    max_chunks: Optional[int] = None
    debug_mode: bool = False

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "HierarchicalChunkerConfig":
        """Create config from dictionary input."""
        if not raw:
            return cls()
        valid_keys = {f.name for f in fields(cls)}
        filtered = {key: value for key, value in raw.items() if key in valid_keys}
        return cls(**filtered)

    def validate(self) -> None:
        """Validate basic configuration constraints."""
        if not self.chunk_sizes:
            raise ValueError("chunk_sizes must not be empty")
        if any(size <= 0 for size in self.chunk_sizes):
            raise ValueError("all chunk_sizes must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if self.min_chunk_chars <= 0:
            raise ValueError("min_chunk_chars must be positive")
        if self.max_chunk_chars and self.max_chunk_chars < self.min_chunk_chars:
            raise ValueError("max_chunk_chars must be >= min_chunk_chars")


@dataclass
class HierarchicalChunk:
    """Structured representation of a hierarchical chunk with parent-child relationships."""

    chunk_id: str
    sequence: int
    text: str
    start: int
    end: int
    depth_level: int  # 0=root, 1=section, 2=paragraph
    parent_chunk_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Convert chunk to serializable dictionary."""
        return {
            "id": self.chunk_id,
            "sequence": self.sequence,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "depth_level": self.depth_level,
            "parent_chunk_id": self.parent_chunk_id,
            "metadata": self.metadata,
        }


class LlamaHierarchicalChunker:
    """Hierarchical chunker for recursive document splitting with parent-child relationships.

    Uses llama_index HierarchicalNodeParser to create multi-level chunks preserving
    document structure (policy > section > paragraph hierarchy).

    Examples:
        >>> config = HierarchicalChunkerConfig(chunk_sizes=[2048, 512, 128])
        >>> chunker = LlamaHierarchicalChunker(config=config, require_llama_index=False)
        >>> chunks = chunker.chunk(
        ...     "# Policy\\n\\n## Section 1\\nDetails about the policy.\\n\\n## Section 2\\nMore details.",
        ...     metadata={"document_id": "doc-123", "source_id": "src-1"},
        ... )
        >>> len(chunks) > 0
        True
        >>> any(chunk.depth_level == 0 for chunk in chunks)  # Has root level
        True
    """

    def __init__(
        self,
        config: Optional[HierarchicalChunkerConfig] = None,
        *,
        require_llama_index: bool = False,
    ) -> None:
        self.config = config or HierarchicalChunkerConfig()
        self.config.validate()
        self._llama_available = (
            _LlamaDocument is not None
            and _HierarchicalNodeParser is not None
            and _SentenceSplitter is not None
        )
        if require_llama_index and not self._llama_available:
            raise ChunkingDependencyError("llama_index is not installed")

    def chunk(
        self,
        text: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        document_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[HierarchicalChunk]:
        """Split raw text into hierarchical chunks with parent-child relationships.

        Args:
            text: Normalized document text.
            metadata: Runtime metadata (e.g., pipeline run id).
            document_metadata: Source metadata (e.g., file name, doc_type).

        Returns:
            List of `HierarchicalChunk` objects with multi-level hierarchy.
        """
        normalized_text = (text or "").strip()
        if not normalized_text:
            return []

        merged_metadata = dict(document_metadata or {})
        if metadata:
            merged_metadata.update(metadata)

        start_time = time.perf_counter()
        try:
            if self._llama_available:
                chunks = self._chunk_with_hierarchical_parser(normalized_text, merged_metadata)
            else:
                if self.config.preserve_hierarchy:
                    logger.warning(
                        "Hierarchical chunking requested but llama_index is unavailable; using fallback splitter.",
                    )
                chunks = self._chunk_with_fallback(normalized_text, merged_metadata)
        except HierarchicalChunkingServiceError:
            raise
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("Unexpected hierarchical chunking failure: %s", exc)
            raise HierarchicalChunkingServiceError(str(exc)) from exc

        duration = time.perf_counter() - start_time
        if self.config.debug_mode:
            logger.debug(
                "Chunked text into %s hierarchical chunks (%s chars) in %.3fs",
                len(chunks),
                len(normalized_text),
                duration,
            )
        return chunks

    def chunk_batch(
        self,
        texts: Iterable[Tuple[str, Dict[str, Any]]],
    ) -> List[List[HierarchicalChunk]]:
        """Chunk a batch of (text, metadata) tuples."""
        results: List[List[HierarchicalChunk]] = []
        for text, metadata in texts:
            results.append(self.chunk(text, metadata=metadata))
        return results

    def _chunk_with_hierarchical_parser(
        self,
        text: str,
        metadata: Dict[str, Any],
    ) -> List[HierarchicalChunk]:
        if not self._llama_available or _LlamaDocument is None or _HierarchicalNodeParser is None:
            raise ChunkingDependencyError("llama_index is not available")

        document = _LlamaDocument(text=text, metadata=metadata)

        try:
            # Create hierarchical parser with configured chunk sizes
            hierarchical_parser = _HierarchicalNodeParser.from_defaults(
                chunk_sizes=self.config.chunk_sizes,
            )
            nodes = hierarchical_parser.get_nodes_from_documents([document])
        except Exception as exc:  # pragma: no cover - hierarchical parser optional
            logger.warning("Hierarchical parser failed (%s); falling back to sentence splitter.", exc)
            return self._chunk_with_fallback(text, metadata)

        if not nodes:
            return self._chunk_with_fallback(text, metadata)

        return self._nodes_to_hierarchical_chunks(nodes, text, metadata)

    def _chunk_with_fallback(
        self,
        text: str,
        metadata: Dict[str, Any],
    ) -> List[HierarchicalChunk]:
        """Simple fallback splitter with single-level hierarchy."""
        # Use the largest chunk size as fallback
        chunk_size = max(self.config.chunk_sizes) if self.config.chunk_sizes else 1024
        overlap = min(self.config.chunk_overlap, chunk_size - 1)
        step = max(chunk_size - overlap, 1)

        chunks: List[HierarchicalChunk] = []
        seq = 0
        index = 0
        text_length = len(text)

        while index < text_length:
            end_index = min(index + chunk_size, text_length)
            window = text[index:end_index]
            leading_ws = len(window) - len(window.lstrip())
            trailing_ws = len(window) - len(window.rstrip())

            start = index + leading_ws
            end = end_index - trailing_ws if trailing_ws else end_index
            chunk_text = window[leading_ws : len(window) - trailing_ws if trailing_ws else len(window)]
            chunk_text = chunk_text.strip()

            if not chunk_text:
                index = end_index
                continue

            if self.config.max_chunk_chars and len(chunk_text) > self.config.max_chunk_chars:
                chunk_text = chunk_text[: self.config.max_chunk_chars].rstrip()
                end = start + len(chunk_text)

            if len(chunk_text) < self.config.min_chunk_chars and end_index < text_length:
                index = max(end_index - overlap, index + step)
                continue

            chunk_metadata = self._filter_metadata(metadata)
            chunk = HierarchicalChunk(
                chunk_id=str(uuid4()),
                sequence=seq,
                text=chunk_text,
                start=start,
                end=end,
                depth_level=0,  # Fallback has no hierarchy
                parent_chunk_id=None,
                metadata=chunk_metadata,
            )
            chunks.append(chunk)
            seq += 1

            if self.config.max_chunks and len(chunks) >= self.config.max_chunks:
                break

            if end_index >= text_length:
                break
            index = max(end_index - overlap, index + step)

        return chunks

    def _nodes_to_hierarchical_chunks(
        self,
        nodes: Sequence[Any],
        original_text: str,
        metadata: Dict[str, Any],
    ) -> List[HierarchicalChunk]:
        """Convert llama_index TextNodes to HierarchicalChunks with parent-child tracking."""
        filtered_metadata = self._filter_metadata(metadata)
        chunks: List[HierarchicalChunk] = []

        # Build mapping of node IDs to chunk IDs for parent tracking
        node_id_to_chunk_id: Dict[str, str] = {}

        for sequence, node in enumerate(nodes):
            content = self._extract_node_content(node)
            if not content:
                continue

            content = self._apply_max_length(content)

            # Calculate character offsets (best effort)
            start = original_text.find(content)
            if start == -1:
                start = 0
            end = start + len(content)

            # Extract hierarchy information
            chunk_id = str(uuid4())
            node_id = getattr(node, "node_id", None) or getattr(node, "id_", None)
            parent_node_id = getattr(node, "parent_node_id", None) or getattr(node, "ref_doc_id", None)

            # Track chunk ID for this node
            if node_id:
                node_id_to_chunk_id[node_id] = chunk_id

            # Resolve parent chunk ID
            parent_chunk_id = node_id_to_chunk_id.get(parent_node_id) if parent_node_id else None

            # Determine depth level from node relationships
            depth_level = self._calculate_depth_level(node, nodes, sequence)

            # Build chunk metadata
            chunk_metadata = filtered_metadata.copy()
            node_metadata = getattr(node, "metadata", None)
            if isinstance(node_metadata, dict):
                for key in self.config.metadata_passthrough_fields:
                    if key in node_metadata:
                        chunk_metadata.setdefault(key, node_metadata[key])

            # Extract heading path and section from node metadata if available
            if node_metadata:
                if "heading_path" in node_metadata:
                    chunk_metadata["heading_path"] = node_metadata["heading_path"]
                if "section" in node_metadata:
                    chunk_metadata["section"] = node_metadata["section"]

            chunks.append(
                HierarchicalChunk(
                    chunk_id=chunk_id,
                    sequence=sequence,
                    text=content,
                    start=start,
                    end=end,
                    depth_level=depth_level,
                    parent_chunk_id=parent_chunk_id,
                    metadata=chunk_metadata,
                ),
            )

            if self.config.max_chunks and len(chunks) >= self.config.max_chunks:
                break

        if not chunks:
            return self._chunk_with_fallback(original_text, metadata)

        return chunks

    def _calculate_depth_level(self, node: Any, all_nodes: Sequence[Any], sequence: int) -> int:
        """Calculate depth level based on chunk size tier.

        HierarchicalNodeParser creates nodes at different levels based on chunk_sizes.
        Level 0 = largest chunks (root), Level N = smallest chunks (leaves).
        """
        node_text_len = len(self._extract_node_content(node))

        # Map text length to configured chunk sizes to determine depth
        for depth, chunk_size in enumerate(sorted(self.config.chunk_sizes, reverse=True)):
            # Allow 10% variance for overlap
            if node_text_len >= chunk_size * 0.9:
                return depth

        # If smaller than all configured sizes, it's the deepest level
        return len(self.config.chunk_sizes) - 1

    def _apply_max_length(self, content: str) -> str:
        if self.config.max_chunk_chars and len(content) > self.config.max_chunk_chars:
            return content[: self.config.max_chunk_chars].rstrip()
        return content

    @staticmethod
    def _extract_node_content(node: Any) -> str:
        if hasattr(node, "get_content"):
            try:
                content = node.get_content()
                if isinstance(content, str) and content.strip():
                    return content.strip()
            except Exception:
                pass

        text_attr = getattr(node, "text", None)
        if isinstance(text_attr, str) and text_attr.strip():
            return text_attr.strip()

        content_attr = getattr(node, "content", None)
        if isinstance(content_attr, str) and content_attr.strip():
            return content_attr.strip()

        return ""

    def _filter_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not metadata:
            return {}
        if not self.config.metadata_passthrough_fields:
            return dict(metadata)
        filtered = {
            key: value for key, value in metadata.items() if key in self.config.metadata_passthrough_fields
        }
        return filtered or dict(metadata)
