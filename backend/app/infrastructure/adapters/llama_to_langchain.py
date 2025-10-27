"""Conversion adapter for llama_index to LangChain interoperability (Task 2.1.3).

This module enables llama_index chunks to be used with LangChain retrieval pipelines,
particularly for the Weaviate vector store and RetrievalQA chains.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from langchain.schema import Document as LangChainDocument
except ImportError:  # pragma: no cover
    LangChainDocument = None  # type: ignore

try:
    from llama_index.core.schema import TextNode
except ImportError:  # pragma: no cover
    TextNode = None  # type: ignore

from app.domain.entities.chunk import Chunk
from app.infrastructure.external_services.ml_services.chunking_services.llama_hierarchical_chunker import (
    HierarchicalChunk,
)


def convert_llama_nodes_to_langchain(nodes: List[Any]) -> List[Any]:
    """
    Convert llama_index TextNodes to LangChain Documents.

    Preserves hierarchical metadata:
    - heading_path: List of heading breadcrumbs
    - section: Section identifier
    - parent_id: Parent chunk reference (from ref_doc_id)
    - depth_level: Hierarchy level (0=root, 1=section, 2=paragraph)

    Args:
        nodes: List of llama_index TextNode objects

    Returns:
        List of LangChain Document objects

    Raises:
        ImportError: If langchain or llama_index are not installed

    Examples:
        >>> from llama_index.core.schema import TextNode
        >>> node = TextNode(
        ...     text="Sample text",
        ...     metadata={"heading_path": ["Policy", "Section 1"], "section": "1"}
        ... )
        >>> docs = convert_llama_nodes_to_langchain([node])
        >>> len(docs)
        1
        >>> docs[0].metadata["heading_path"]
        ['Policy', 'Section 1']
    """
    if LangChainDocument is None:
        raise ImportError("langchain is required for this conversion. Install with: pip install langchain")

    if TextNode is None:
        raise ImportError("llama_index is required for this conversion. Install with: pip install llama-index")

    langchain_docs: List[Any] = []

    for node in nodes:
        # Extract text content
        text = _extract_node_text(node)
        if not text:
            continue

        # Extract node identifiers
        node_id = getattr(node, "node_id", None) or getattr(node, "id_", None)
        parent_id = getattr(node, "parent_node_id", None) or getattr(node, "ref_doc_id", None)

        # Build metadata
        node_metadata = getattr(node, "metadata", {}) or {}
        metadata: Dict[str, Any] = {
            "chunk_id": str(node_id) if node_id else None,
            "parent_id": str(parent_id) if parent_id else None,
            **node_metadata,
        }

        # Ensure standard fields are present
        if "heading_path" not in metadata:
            metadata["heading_path"] = []
        if "section" not in metadata:
            metadata["section"] = None
        if "depth_level" not in metadata:
            metadata["depth_level"] = 0

        # Create LangChain Document
        doc = LangChainDocument(page_content=text, metadata=metadata)
        langchain_docs.append(doc)

    return langchain_docs


def convert_hierarchical_chunks_to_langchain(chunks: List[HierarchicalChunk]) -> List[Any]:
    """
    Convert HierarchicalChunk objects to LangChain Documents.

    This is a higher-level conversion that works directly with our
    HierarchicalChunk entities rather than raw llama_index nodes.

    Args:
        chunks: List of HierarchicalChunk objects

    Returns:
        List of LangChain Document objects

    Raises:
        ImportError: If langchain is not installed

    Examples:
        >>> from app.infrastructure.external_services.ml_services.chunking_services import HierarchicalChunk
        >>> chunk = HierarchicalChunk(
        ...     chunk_id="chunk-1",
        ...     sequence=0,
        ...     text="Sample text",
        ...     start=0,
        ...     end=11,
        ...     depth_level=0,
        ...     parent_chunk_id=None,
        ...     metadata={"section": "root"}
        ... )
        >>> docs = convert_hierarchical_chunks_to_langchain([chunk])
        >>> len(docs)
        1
        >>> docs[0].metadata["depth_level"]
        0
    """
    if LangChainDocument is None:
        raise ImportError("langchain is required for this conversion. Install with: pip install langchain")

    langchain_docs: List[Any] = []

    for chunk in chunks:
        metadata = {
            "chunk_id": chunk.chunk_id,
            "sequence": chunk.sequence,
            "start": chunk.start,
            "end": chunk.end,
            "depth_level": chunk.depth_level,
            "parent_chunk_id": chunk.parent_chunk_id,
            **chunk.metadata,
        }

        doc = LangChainDocument(page_content=chunk.text, metadata=metadata)
        langchain_docs.append(doc)

    return langchain_docs


def convert_domain_chunks_to_langchain(chunks: List[Chunk]) -> List[Any]:
    """
    Convert domain Chunk entities to LangChain Documents.

    This conversion is useful when working with chunks retrieved from
    the repository that need to be used in LangChain retrieval pipelines.

    Args:
        chunks: List of domain Chunk objects

    Returns:
        List of LangChain Document objects

    Raises:
        ImportError: If langchain is not installed

    Examples:
        >>> from app.domain.entities.chunk import Chunk
        >>> from app.domain.value_objects.chunk_metadata import ChunkMetadata
        >>> from uuid import uuid4
        >>> chunk = Chunk(
        ...     chunk_id="chunk-1",
        ...     document_id=uuid4(),
        ...     text="Sample text",
        ...     metadata=ChunkMetadata(
        ...         sequence=0,
        ...         start=0,
        ...         end=11,
        ...         extra={"depth_level": 0}
        ...     )
        ... )
        >>> docs = convert_domain_chunks_to_langchain([chunk])
        >>> len(docs)
        1
    """
    if LangChainDocument is None:
        raise ImportError("langchain is required for this conversion. Install with: pip install langchain")

    langchain_docs: List[Any] = []

    for chunk in chunks:
        metadata = {
            "chunk_id": chunk.chunk_id,
            "document_id": str(chunk.document_id),
            "sequence": chunk.metadata.sequence,
            "start": chunk.metadata.start,
            "end": chunk.metadata.end,
            "length": chunk.metadata.length,
        }

        # Add optional metadata fields
        if chunk.metadata.token_count is not None:
            metadata["token_count"] = chunk.metadata.token_count
        if chunk.metadata.section:
            metadata["section"] = chunk.metadata.section
        if chunk.metadata.heading_path:
            metadata["heading_path"] = chunk.metadata.heading_path
        if chunk.metadata.page_number is not None:
            metadata["page_number"] = chunk.metadata.page_number
        if chunk.metadata.source_type:
            metadata["source_type"] = chunk.metadata.source_type

        # Add extra metadata (includes depth_level, parent_id for hierarchical chunks)
        if chunk.metadata.extra:
            metadata.update(chunk.metadata.extra)

        # Add embedding and score if present
        if chunk.embedding_vector_id:
            metadata["embedding_vector_id"] = chunk.embedding_vector_id
        if chunk.score is not None:
            metadata["similarity_score"] = chunk.score

        doc = LangChainDocument(page_content=chunk.text, metadata=metadata)
        langchain_docs.append(doc)

    return langchain_docs


def _extract_node_text(node: Any) -> Optional[str]:
    """Extract text content from llama_index node."""
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

    return None
