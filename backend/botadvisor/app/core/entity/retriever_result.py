"""
Retriever Result Entity

Result model for retriever operations.
"""

from dataclasses import dataclass

from .document import Document
from .document_metadata import DocumentMetadata

@dataclass
class RetrieverResult:
    """
    Result model for retriever operations.

    Attributes:
        content: Retrieved text content
        score: Similarity/relevance score (0-1)
        metadata: Document and chunk metadata for citations
        source: Source document reference
    """

    content: str
    score: float
    metadata: DocumentMetadata
    source: Document
