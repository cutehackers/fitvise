"""Vector store services for RAG system."""

# Import weaviate_schema (no external dependencies)
from .weaviate_schema import WeaviateSchema

# Import weaviate_client only if weaviate is available
try:
    from .weaviate_client import WeaviateClient
    __all__ = [
        "WeaviateClient",
        "WeaviateSchema",
    ]
except ImportError:
    # Weaviate client not available (missing weaviate package)
    __all__ = [
        "WeaviateSchema",
    ]
