"""LlamaIndex-backed chunking service with semantic + fallback strategies."""

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
    from llama_index.core.node_parser import SentenceSplitter as _SentenceSplitter  # type: ignore
    try:
        from llama_index.core.node_parser import (  # type: ignore
            SemanticSplitterNodeParser as _SemanticSplitter,
        )
    except Exception:  # pragma: no cover - semantic splitter optional
        _SemanticSplitter = None  # type: ignore
except Exception:  # pragma: no cover - llama_index not installed
    _LlamaDocument = None  # type: ignore
    _SentenceSplitter = None  # type: ignore
    _SemanticSplitter = None  # type: ignore

class ChunkingServiceError(ChunkGenerationError):
    """Raised when chunking fails due to unexpected runtime errors."""


@dataclass
class LlamaIndexChunkerConfig:
    """Runtime configuration for semantic chunking."""

    chunk_size: int = 1024
    chunk_overlap: int = 128
    separators: Sequence[str] = ("\n\n", "\n", " ")
    enable_semantic: bool = True
    semantic_breakpoint_threshold: float = 0.85
    min_chunk_chars: int = 80
    max_chunk_chars: Optional[int] = None
    metadata_passthrough_fields: Sequence[str] = ("document_id", "source_id", "file_name", "doc_type")
    max_chunks: Optional[int] = None
    debug_mode: bool = False

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "LlamaIndexChunkerConfig":
        """Create config from dictionary input."""
        if not raw:
            return cls()
        valid_keys = {f.name for f in fields(cls)}
        filtered = {key: value for key, value in raw.items() if key in valid_keys}
        return cls(**filtered)

    def validate(self) -> None:
        """Validate basic configuration constraints."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.min_chunk_chars <= 0:
            raise ValueError("min_chunk_chars must be positive")


@dataclass
class SemanticChunk:
    """Structured representation of a single chunk."""

    chunk_id: str
    sequence: int
    text: str
    start: int
    end: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Convert chunk to serializable dictionary."""
        return {
            "id": self.chunk_id,
            "sequence": self.sequence,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "metadata": self.metadata,
        }


class LlamaIndexChunker:
    """Semantic chunker bridging domain documents and embedding pipelines.

    Examples:
        >>> config = LlamaIndexChunkerConfig(chunk_size=512, chunk_overlap=64)
        >>> chunker = LlamaIndexChunker(config=config, require_llama_index=False)
        >>> chunks = chunker.chunk(
        ...     "Section 1\\nDetails about the policy.\\n\\nSection 2\\nMore details.",
        ...     metadata={"document_id": "doc-123", "source_id": "src-1"},
        ... )
        >>> len(chunks) > 0
        True
        >>> chunks[0].metadata["document_id"]
        'doc-123'
    """

    def __init__(
        self,
        config: Optional[LlamaIndexChunkerConfig] = None,
        *,
        require_llama_index: bool = False,
    ) -> None:
        self.config = config or LlamaIndexChunkerConfig()
        self.config.validate()
        self._llama_available = _LlamaDocument is not None and _SentenceSplitter is not None
        if require_llama_index and not self._llama_available:
            raise ChunkingDependencyError("llama_index is not installed")

    def chunk(
        self,
        text: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        document_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SemanticChunk]:
        """Split raw text into semantic chunks.

        Args:
            text: Normalized document text.
            metadata: Runtime metadata (e.g., pipeline run id).
            document_metadata: Source metadata (e.g., file name, doc_type).

        Returns:
            List of `SemanticChunk` objects ready for downstream ingestion.
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
                chunks = self._chunk_with_llama_index(normalized_text, merged_metadata)
            else:
                if self.config.enable_semantic:
                    logger.warning(
                        "Semantic chunking requested but llama_index is unavailable; using fallback splitter.",
                    )
                chunks = self._chunk_with_fallback(normalized_text, merged_metadata)
        except ChunkingServiceError:
            raise
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("Unexpected chunking failure: %s", exc)
            raise ChunkingServiceError(str(exc)) from exc

        duration = time.perf_counter() - start_time
        if self.config.debug_mode:
            logger.debug(
                "Chunked text into %s chunks (%s chars) in %.3fs",
                len(chunks),
                len(normalized_text),
                duration,
            )
        return chunks

    def chunk_batch(
        self,
        texts: Iterable[Tuple[str, Dict[str, Any]]],
    ) -> List[List[SemanticChunk]]:
        """Chunk a batch of (text, metadata) tuples."""
        results: List[List[SemanticChunk]] = []
        for text, metadata in texts:
            results.append(self.chunk(text, metadata=metadata))
        return results

    def _chunk_with_llama_index(
        self,
        text: str,
        metadata: Dict[str, Any],
    ) -> List[SemanticChunk]:
        if not self._llama_available or _LlamaDocument is None or _SentenceSplitter is None:
            raise ChunkingDependencyError("llama_index is not available")

        nodes: List[Any] = []
        document = _LlamaDocument(text=text, metadata=metadata)

        if self.config.enable_semantic and _SemanticSplitter is not None:
            try:
                # SemanticSplitterNodeParser expects breakpoint_percentile_threshold as an integer (0-100)
                # Convert from float (0-1) to integer (0-100) if needed
                threshold = self.config.semantic_breakpoint_threshold
                threshold_int = int(threshold * 100) if threshold <= 1.0 else int(threshold)

                semantic_parser = _SemanticSplitter.from_defaults(  # type: ignore[attr-defined]
                    breakpoint_percentile_threshold=threshold_int,
                )
                nodes = semantic_parser.get_nodes_from_documents([document])
            except Exception as exc:  # pragma: no cover - semantic splitter optional
                logger.warning("Semantic splitter failed (%s); falling back to sentence splitter.", exc)
                nodes = []

        if not nodes:
            try:
                sentence_splitter = _SentenceSplitter(
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap,
                )
                nodes = sentence_splitter.get_nodes_from_documents([document])
            except Exception as exc:  # pragma: no cover
                logger.warning("Sentence splitter failed (%s); using fallback splitter.", exc)
                return self._chunk_with_fallback(text, metadata)

        return self._nodes_to_chunks(nodes, text, metadata)

    def _chunk_with_fallback(
        self,
        text: str,
        metadata: Dict[str, Any],
    ) -> List[SemanticChunk]:
        chunk_size = self.config.chunk_size
        overlap = min(self.config.chunk_overlap, chunk_size - 1)
        step = max(chunk_size - overlap, 1)

        chunks: List[SemanticChunk] = []
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
            chunk = SemanticChunk(
                chunk_id=str(uuid4()),
                sequence=seq,
                text=chunk_text,
                start=start,
                end=end,
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

    def _nodes_to_chunks(
        self,
        nodes: Sequence[Any],
        original_text: str,
        metadata: Dict[str, Any],
    ) -> List[SemanticChunk]:
        filtered_metadata = self._filter_metadata(metadata)
        pointer = 0
        chunks: List[SemanticChunk] = []

        for sequence, node in enumerate(nodes):
            content = self._extract_node_content(node)
            if not content:
                continue

            content = self._apply_max_length(content)

            start = original_text.find(content, pointer)
            if start == -1:
                start = pointer
            end = start + len(content)
            pointer = end

            chunk_metadata = filtered_metadata.copy()
            node_metadata = getattr(node, "metadata", None)
            if isinstance(node_metadata, dict):
                for key in self.config.metadata_passthrough_fields:
                    if key in node_metadata:
                        chunk_metadata.setdefault(key, node_metadata[key])

            chunks.append(
                SemanticChunk(
                    chunk_id=str(uuid4()),
                    sequence=sequence,
                    text=content,
                    start=start,
                    end=end,
                    metadata=chunk_metadata,
                ),
            )

            if self.config.max_chunks and len(chunks) >= self.config.max_chunks:
                break

        if not chunks:
            return self._chunk_with_fallback(original_text, metadata)

        return chunks

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
            key: value
            for key, value in metadata.items()
            if key in self.config.metadata_passthrough_fields
        }
        return filtered or dict(metadata)
