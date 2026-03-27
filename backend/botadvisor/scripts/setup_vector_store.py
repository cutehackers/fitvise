#!/usr/bin/env python3
"""Bootstrap the canonical vector-store collection for BotAdvisor."""

from __future__ import annotations

import argparse
from urllib.parse import urlparse

import weaviate
from weaviate.classes.config import Configure, DataType, Property, VectorDistances

from botadvisor.app.core.config import get_settings
from botadvisor.app.observability.logging import configure_logger, get_logger


DEFAULT_COLLECTION_NAME = "Chunk"
DEFAULT_EMBEDDING_DIMENSION = 384

configure_logger()
logger = get_logger("setup_vector_store")


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for vector-store bootstrap."""
    parser = argparse.ArgumentParser(description="Bootstrap the canonical Weaviate collection for BotAdvisor")
    parser.add_argument("--force", action="store_true", help="Delete the existing collection before re-creating it")
    parser.add_argument("--dimension", type=int, default=None, help="Embedding dimension reported in the bootstrap summary")
    return parser


def resolve_embedding_dimension(requested_dimension: int | None = None) -> int:
    """Resolve the embedding dimension used by the canonical bootstrap flow."""
    return requested_dimension or DEFAULT_EMBEDDING_DIMENSION


def connect_to_weaviate():
    """Create a Weaviate client using canonical runtime settings."""
    settings = get_settings()
    parsed = urlparse(settings.weaviate_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 8080)
    return weaviate.connect_to_local(
        host=host,
        port=port,
        grpc_port=settings.weaviate_grpc_port,
    )


def collection_exists(client, collection_name: str) -> bool:
    """Return whether the configured collection already exists."""
    exists = getattr(client.collections, "exists", None)
    if callable(exists):
        return bool(exists(collection_name))

    try:
        client.collections.get(collection_name)
    except Exception:
        return False
    return True


def create_collection(client, collection_name: str) -> None:
    """Create the canonical collection schema used by retrieval."""
    client.collections.create(
        name=collection_name,
        vectorizer_config=Configure.Vectorizer.none(),
        vector_index_config=Configure.VectorIndex.hnsw(distance_metric=VectorDistances.COSINE),
        properties=[
            Property(name="content", data_type=DataType.TEXT),
            Property(name="doc_id", data_type=DataType.TEXT),
            Property(name="source_id", data_type=DataType.TEXT),
            Property(name="source_url", data_type=DataType.TEXT),
            Property(name="platform", data_type=DataType.TEXT),
            Property(name="section", data_type=DataType.TEXT),
            Property(name="checksum", data_type=DataType.TEXT),
        ],
    )


def setup_vector_store(*, force: bool, dimension: int | None) -> dict[str, object]:
    """Create the canonical vector-store collection when needed."""
    resolved_dimension = resolve_embedding_dimension(dimension)
    collection_name = DEFAULT_COLLECTION_NAME
    client = connect_to_weaviate()

    try:
        if collection_exists(client, collection_name):
            if not force:
                return {
                    "backend": "weaviate",
                    "collection_name": collection_name,
                    "created": False,
                    "force": False,
                    "dimension": resolved_dimension,
                }
            client.collections.delete(collection_name)

        create_collection(client, collection_name)
        return {
            "backend": "weaviate",
            "collection_name": collection_name,
            "created": True,
            "force": force,
            "dimension": resolved_dimension,
        }
    finally:
        client.close()


def main(argv: list[str] | None = None) -> int:
    """Run the vector-store bootstrap CLI."""
    args = create_parser().parse_args(argv)
    summary = setup_vector_store(force=args.force, dimension=args.dimension)
    logger.info("Vector-store bootstrap complete", extra={"summary": summary})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
