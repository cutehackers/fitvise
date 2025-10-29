"""Vector store services for RAG system."""

# Import weaviate_schema (no external dependencies)
from .weaviate_schema import WeaviateSchema, create_chunk_class_schema

# Import weaviate_client only if weaviate is available
try:
    from .weaviate_client import WeaviateClient
    __all__ = [
        "WeaviateClient",
        "WeaviateSchema",
        "create_chunk_class_schema",
    ]
except ImportError:
    # Weaviate client not available (missing weaviate package)
    __all__ = [
        "WeaviateSchema",
        "create_chunk_class_schema",
    ]
