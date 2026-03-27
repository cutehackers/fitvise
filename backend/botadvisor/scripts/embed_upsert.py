#!/usr/bin/env python3
"""
Embedding and Upsert Script

Reads normalized chunks (JSON) produced by ingest.py, generates embeddings,
and idempotently upserts them into a vector store (Weaviate or Chroma).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from botadvisor.app.core.config import get_settings
from botadvisor.app.observability.langfuse import get_tracer
from botadvisor.app.observability.logging import configure_logger, get_logger
from tenacity import retry, stop_after_attempt, wait_exponential

# LlamaIndex imports
try:
    from llama_index.core.schema import TextNode
    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.vector_stores.weaviate import WeaviateVectorStore
    from llama_index.vector_stores.chroma import ChromaVectorStore
    import weaviate
    import chromadb
except ImportError as e:
    print(f"Error importing LlamaIndex components: {e}")
    print("Please ensure you have installed the necessary dependencies.")
    sys.exit(1)

# Configure logging
configure_logger()
logger = get_logger("embed_upsert")
tracer = get_tracer()


class EmbedUpsertScript:
    """
    Script to embed and upsert chunks into a vector store.
    """

    def __init__(self, store_type: str, model_name: str, batch_size: int, collection_name: str, url: str = None):
        """
        Initialize the script.

        Args:
            store_type: 'weaviate' or 'chroma'
            model_name: Embedding model name
            batch_size: Batch size for processing
            collection_name: Name of the collection/class in vector store
            url: URL for Weaviate (optional)
        """
        self.store_type = store_type
        self.model_name = model_name
        self.batch_size = batch_size
        self.collection_name = collection_name
        self.url = url

        self.embed_model = self._init_embedding_model(model_name)
        self.vector_store = self._init_vector_store()
        self.pipeline = self._init_pipeline()

        self.total_nodes_processed = 0
        self.total_batches = 0
        self.successful_batches = 0
        self.failed_batches = 0

    def _init_embedding_model(self, model_name: str):
        """Initialize the embedding model."""
        logger.info(f"Initializing embedding model: {model_name}")
        try:
            # Check if it looks like an Ollama model or HF
            if "ollama" in model_name.lower() or ":" in model_name:
                # Assume format "provider:model" or just "model" for Ollama if simple
                # For simplicity here, if it starts with 'ollama:', use OllamaEmbedding
                if model_name.startswith("ollama:"):
                    pure_name = model_name.split(":", 1)[1]
                    return OllamaEmbedding(model_name=pure_name)

            # Default to HuggingFace
            return HuggingFaceEmbedding(model_name=model_name)
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise

    def _init_vector_store(self):
        """Initialize the vector store client."""
        logger.info(f"Initializing vector store: {self.store_type}")
        try:
            if self.store_type == "weaviate":
                settings = get_settings()
                weaviate_url = self.url or settings.weaviate_url
                parsed = urlparse(weaviate_url)
                host = parsed.hostname or "localhost"
                port = parsed.port or (443 if parsed.scheme == "https" else 8080)
                grpc_port = settings.weaviate_grpc_port

                client = weaviate.connect_to_local(
                    host=host,
                    port=port,
                    grpc_port=grpc_port,
                )
                return WeaviateVectorStore(weaviate_client=client, index_name=self.collection_name)

            elif self.store_type == "chroma":
                # Use a local persistence directory for Chroma
                persist_dir = get_settings().chroma_persist_dir
                client = chromadb.PersistentClient(path=persist_dir)
                collection = client.get_or_create_collection(self.collection_name)
                return ChromaVectorStore(chroma_collection=collection)

            else:
                raise ValueError(f"Unsupported store type: {self.store_type}")

        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise

    def _init_pipeline(self):
        """Initialize the ingestion pipeline."""
        return IngestionPipeline(
            transformations=[self.embed_model],
            vector_store=self.vector_store,
        )

    def _load_chunks(self, file_path: Path) -> List[TextNode]:
        """
        Load chunks from a JSON file and convert to TextNodes.
        """
        nodes = []
        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                logger.warning(f"File {file_path} does not contain a list of chunks.")
                return []

            for item in data:
                if "content" not in item or "metadata" not in item:
                    continue

                # Create TextNode
                # We use the chunk_id as the node id for idempotency (if store supports it)
                node = TextNode(
                    text=item["content"],
                    id_=item.get("chunk_id"),
                    metadata=item.get("metadata", {}),
                )

                # Ensure metadata is flat/valid for vector stores
                # LlamaIndex handles some nesting, but safer to keep simple
                # Ensure excluded keys are managed if needed (excluded_embed_metadata_keys)
                node.excluded_embed_metadata_keys = ["doc_id", "source_id", "source_url", "page", "section"]
                node.excluded_llm_metadata_keys = ["doc_id", "checksum"]

                nodes.append(node)

        except Exception as e:
            logger.error(f"Error loading chunks from {file_path}: {e}")

        return nodes

    def process_file(self, file_path: Path):
        """Process a single JSON file containing chunks."""
        logger.info(f"Processing file: {file_path}")

        nodes = self._load_chunks(file_path)
        if not nodes:
            return

        total_file_nodes = len(nodes)

        # Process in batches
        for i in range(0, total_file_nodes, self.batch_size):
            batch_nodes = nodes[i : i + self.batch_size]
            self._process_batch(batch_nodes)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), reraise=True)
    def _run_pipeline_with_retry(self, nodes: List[TextNode]):
        """Run pipeline with retry logic."""
        self.pipeline.run(nodes=nodes)

    def _process_batch(self, nodes: List[TextNode]):
        """Process a batch of nodes through the pipeline."""
        if not nodes:
            return

        self.total_batches += 1

        trace = None
        if tracer.is_enabled():
            trace = tracer.trace(name="embed_batch", trace_type="embedding")

        try:
            # Run pipeline with retry
            self._run_pipeline_with_retry(nodes)

            self.total_nodes_processed += len(nodes)
            self.successful_batches += 1
            logger.info(f"Processed batch of {len(nodes)} nodes. Total: {self.total_nodes_processed}")

            if trace:
                trace.update(status="success", output={"count": len(nodes)})

        except Exception as e:
            self.failed_batches += 1
            logger.error(f"Failed to process batch after retries: {e}")
            if trace:
                trace.update(status="error", output={"error": str(e)})

    def run(self, input_path: str):
        """Run the embedding and upsert process."""
        path = Path(input_path)

        files_to_process = []
        if path.is_file():
            files_to_process.append(path)
        elif path.is_dir():
            files_to_process.extend(path.glob("**/*.json"))
        else:
            logger.error(f"Input path not found: {input_path}")
            return

        logger.info(f"Found {len(files_to_process)} files to process.")

        for file_path in files_to_process:
            self.process_file(file_path)

        summary = {
            "files_found": len(files_to_process),
            "nodes_processed": self.total_nodes_processed,
            "total_batches": self.total_batches,
            "successful_batches": self.successful_batches,
            "failed_batches": self.failed_batches,
        }

        logger.info("Embedding and upsert complete.")
        logger.info(f"Summary: {summary}")
        return summary


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the embed/upsert script."""
    parser = argparse.ArgumentParser(description="Embed and Upsert Script")
    parser.add_argument(
        "--input",
        "--in",
        "-i",
        dest="input_path",
        required=True,
        help="Input file or directory (containing JSON chunks)",
    )
    parser.add_argument("--store", "-s", choices=["weaviate", "chroma"], default="chroma", help="Vector store type")
    parser.add_argument(
        "--model",
        "-m",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name (HuggingFace or ollama:model)",
    )
    parser.add_argument("--batch-size", "-b", type=int, default=100, help="Batch size")
    parser.add_argument("--collection", "-c", default="BotAdvisorDocs", help="Collection/Class name in vector store")
    parser.add_argument("--url", "-u", help="Weaviate URL (overrides WEAVIATE_URL env var)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = create_parser().parse_args(argv)

    if args.verbose:
        configure_logger(level="DEBUG")

    script = EmbedUpsertScript(
        store_type=args.store,
        model_name=args.model,
        batch_size=args.batch_size,
        collection_name=args.collection,
        url=args.url,
    )

    summary = script.run(args.input_path)
    return 0 if summary["failed_batches"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
