"""Initialize Weaviate schema for RAG system (Task 2.3.2).

This script creates the Chunk class schema in Weaviate with all metadata fields.
Run this after starting Weaviate to initialize the database.

Usage:
    python scripts/setup_weaviate_schema.py [--force] [--dimension DIMENSION]

Options:
    --force: Delete existing schema and recreate
    --dimension: Embedding vector dimension (default: read from config)
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import weaviate
from weaviate.exceptions import WeaviateBaseError

from app.config.vector_stores.weaviate_config import WeaviateConfig
from app.config.ml_models.embedding_model_configs import EmbeddingModelConfig
from app.infrastructure.external_services.vector_stores.weaviate_schema import (
    WeaviateSchema,
    SCHEMA_VERSION,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def init_schema(force: bool = False, dimension: int = None) -> None:
    """Initialize Weaviate schema.

    Args:
        force: If True, delete existing schema and recreate
        dimension: Embedding vector dimension (uses config default if not specified)
    """
    logger.info("Initializing Weaviate schema (Task 2.3.2)")
    logger.info(f"Schema version: {SCHEMA_VERSION}")

    # Load configuration
    config = WeaviateConfig.for_local_development()
    logger.info(f"Connecting to Weaviate at {config.get_url()}")

    # Get dimension from config if not provided
    if dimension is None:
        embedding_config = EmbeddingModelConfig.default()
        dimension = embedding_config.model_dimension
        logger.info(f"Using dimension from config: {dimension} ({embedding_config.model_name})")
    else:
        logger.info(f"Using dimension from CLI argument: {dimension}")

    # Create client
    try:
        client = weaviate.Client(url=config.get_url(), timeout_config=(5, 30))
        logger.info("Connected to Weaviate successfully")
    except Exception as e:
        logger.error(f"Failed to connect to Weaviate: {e}")
        logger.error("Make sure Weaviate is running (docker-compose up weaviate)")
        sys.exit(1)

    # Create schema manager
    schema_manager = WeaviateSchema(client)

    try:
        # Check if Chunk class exists
        exists = await schema_manager.class_exists("Chunk")

        if exists:
            if force:
                logger.warning("Chunk class exists, deleting (--force mode)")
                await schema_manager.delete_class("Chunk")
                logger.info("Deleted existing Chunk class")
            else:
                logger.info("Chunk class already exists, skipping creation")
                logger.info("Use --force to delete and recreate")
                return

        # Create Chunk class
        logger.info(f"Creating Chunk class with {dimension}-dimensional vectors...")
        await schema_manager.create_chunk_class(
            dimension=dimension,
            distance_metric="cosine",
            description="Document chunks with embeddings for RAG system (Task 2.3.2)",
        )
        logger.info("✅ Chunk class created successfully")

        # Verify creation
        class_def = await schema_manager.get_class("Chunk")
        if class_def:
            num_properties = len(class_def.get("properties", []))
            logger.info(f"Schema has {num_properties} properties")
            logger.info(f"Vector index: {class_def.get('vectorIndexConfig', {}).get('distance')}")
        else:
            logger.error("Failed to verify schema creation")
            sys.exit(1)

        logger.info("✅ Schema initialization complete")

    except WeaviateBaseError as e:
        logger.error(f"Weaviate error during schema initialization: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Initialize Weaviate schema for RAG system"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing schema and recreate",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=None,
        help="Embedding vector dimension (default: read from config)",
    )
    args = parser.parse_args()

    # Run async initialization
    asyncio.run(init_schema(force=args.force, dimension=args.dimension))


if __name__ == "__main__":
    main()
