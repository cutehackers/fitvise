"""Retrieval boundary for hybrid-search-aware RAG services."""

from .config import RetrievalConfig
from .factory import create_hybrid_retriever
from .langchain_adapter import LangChainRetrieverAdapter

__all__ = [
    "LangChainRetrieverAdapter",
    "RetrievalConfig",
    "create_hybrid_retriever",
]
