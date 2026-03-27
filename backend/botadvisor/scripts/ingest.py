#!/usr/bin/env python3
"""
Simple Ingestion Script for Testing

Simplified version of the Docling ingestion script that focuses on core functionality
and uses only local storage for testing purposes.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Any

from botadvisor.app.core.config import get_settings
from botadvisor.app.core.entity.chunk import Chunk
from botadvisor.app.core.entity.document import Document
from botadvisor.app.ingestion.chunking import create_chunks
from botadvisor.app.ingestion.files import detect_mime_type, extract_source_metadata
from botadvisor.app.ingestion.readers import DOCLING_AVAILABLE, get_reader_for_file, read_text_with_reader
from botadvisor.app.observability.langfuse import get_tracer
from botadvisor.app.observability.logging import get_logger, configure_logger
from botadvisor.app.storage.factory import get_storage_backend

# Configure logging
configure_logger()
logger = get_logger("ingest")
tracer = get_tracer()

if DOCLING_AVAILABLE:
    logger.info("Docling imported successfully")
else:
    logger.warning("Docling not available. Using fallback text processing.")


class SimpleIngestionScript:
    """
    Simplified document ingestion script for testing.

    Focuses on core functionality with local storage only.
    """

    def __init__(self):
        """Initialize ingestion script with local storage backend."""
        self.storage_backend = get_storage_backend(get_settings())
        self.duplicates_skipped = 0
        self.documents_processed = 0
        self.chunks_generated = 0

    def _create_document(self, file_path: Path, platform: str, content: bytes) -> Document:
        """
        Create Document entity from file and content.

        Args:
            file_path: Path to source file
            platform: Source platform
            content: Raw content bytes

        Returns:
            Document entity with computed checksum
        """
        metadata = extract_source_metadata(file_path, platform)

        return Document.create(
            source_id=metadata["source_id"],
            platform=platform,
            source_url=metadata["source_url"],
            content=content,
            mime_type=detect_mime_type(file_path),
        )

    def _parse_document(self, file_path: Path, platform: str) -> Optional[Tuple[Document, str]]:
        """
        Parse document using Docling and create Document entity.

        Args:
            file_path: Path to document file
            platform: Source platform

        Returns:
            Tuple of (Document, raw_text) if successful, None otherwise
        """
        try:
            # Read file content
            with file_path.open("rb") as f:
                content = f.read()

            # Create document entity
            document = self._create_document(file_path, platform, content)

            # Check if document already exists (deduplication)
            if self.storage_backend.exists(document.checksum):
                logger.info(
                    f"Skipping duplicate document: {file_path.name} (checksum: {document.checksum[:8]}...)",
                    extra={"checksum": document.checksum, "file": str(file_path)},
                )
                self.duplicates_skipped += 1
                return None

            # Get appropriate reader
            reader = get_reader_for_file(file_path, logger)
            if not reader:
                return None

            # Parse document content
            raw_text = read_text_with_reader(reader, content)

            return document, raw_text

        except Exception as e:
            logger.error(
                f"Failed to parse document {file_path.name}: {str(e)}", extra={"file": str(file_path), "error": str(e)}
            )
            return None

    def _store_document(self, document: Document, content: bytes) -> Any:
        """
        Store document content using storage backend.

        Args:
            document: Document entity
            content: Raw content bytes

        Returns:
            StorageArtifact with storage details
        """
        try:
            artifact = self.storage_backend.save(content, document)
            logger.info(
                f"Stored document: {document.source_id}",
                extra={
                    "document_id": document.id,
                    "checksum": document.checksum,
                    "storage_uri": artifact.uri,
                    "size_bytes": artifact.size_bytes,
                    "already_existed": artifact.already_existed,
                },
            )
            return artifact
        except Exception as e:
            logger.error(
                f"Failed to store document {document.id}: {str(e)}", extra={"document_id": document.id, "error": str(e)}
            )
            raise

    def _save_chunks(self, chunks: List[Chunk], output_dir: Path, platform: str):
        """
        Save chunks to output directory as JSON files.

        Args:
            chunks: List of Chunk entities
            output_dir: Output directory path
            platform: Source platform
        """
        # Create platform-specific output directory
        platform_dir = output_dir / platform
        platform_dir.mkdir(parents=True, exist_ok=True)

        # Save chunks as JSON
        chunks_data = [chunk.to_dict() for chunk in chunks]
        output_file = platform_dir / f"chunks_{int(datetime.now().timestamp())}.json"

        with output_file.open("w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Saved {len(chunks)} chunks to {output_file}",
            extra={"chunk_count": len(chunks), "output_file": str(output_file)},
        )

    def process_file(self, file_path: Path, platform: str, output_dir: Path):
        """
        Process single file through ingestion pipeline.

        Args:
            file_path: Path to file to process
            platform: Source platform
            output_dir: Output directory for chunks
        """
        logger.info(f"Processing file: {file_path.name}", extra={"file": str(file_path)})

        # Start LangFuse trace for this document
        trace = None
        if tracer.is_enabled():
            trace = tracer.trace(
                name=f"ingest_{file_path.name}",
                trace_type="ingestion",
                metadata={"file_name": file_path.name, "platform": platform, "file_size": file_path.stat().st_size},
            )

        try:
            # Parse document
            result = self._parse_document(file_path, platform)
            if not result:
                return  # Document was skipped or failed to parse

            document, raw_text = result

            # Store raw document
            with file_path.open("rb") as f:
                content = f.read()

            artifact = self._store_document(document, content)

            # Chunk text
            chunks = create_chunks(raw_text, document=document)
            self.chunks_generated += len(chunks)

            # Save chunks
            self._save_chunks(chunks, output_dir, platform)

            self.documents_processed += 1

            logger.info(
                f"Successfully processed {file_path.name}: " f"{len(chunks)} chunks, {artifact.size_bytes} bytes",
                extra={
                    "file": str(file_path),
                    "chunks": len(chunks),
                    "bytes": artifact.size_bytes,
                    "checksum": document.checksum,
                },
            )

        except Exception as e:
            logger.error(
                f"Failed to process file {file_path.name}: {str(e)}", extra={"file": str(file_path), "error": str(e)}
            )
            if trace:
                trace.update(status="error", output={"error": str(e)})

        finally:
            if trace:
                trace.update(
                    status="completed",
                    output={
                        "documents_processed": self.documents_processed,
                        "chunks_generated": self.chunks_generated,
                        "duplicates_skipped": self.duplicates_skipped,
                    },
                )


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the ingestion script."""
    parser = argparse.ArgumentParser(
        description="BotAdvisor Simple Document Ingestion Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--input", "-i", required=True, help="Input file or directory path")

    parser.add_argument("--out", "-o", default="./data/chunks", help="Output directory for processed chunks")

    parser.add_argument("--platform", "-p", required=True, help="Source platform (e.g., filesystem, web, gdrive)")

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for ingestion script."""
    args = create_parser().parse_args(argv)

    # Configure logging level
    if args.verbose:
        configure_logger(level="DEBUG")
        global logger
        logger = get_logger("ingest")  # Re-initialize with new level

    logger.info(
        "Starting BotAdvisor simple ingestion script",
        extra={
            "input": args.input,
            "output": args.out,
            "platform": args.platform,
            "langfuse_enabled": tracer.is_enabled(),
        },
    )

    # Initialize ingestion script
    ingestor = SimpleIngestionScript()

    # Convert paths
    input_path = Path(args.input)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process input
    if input_path.is_file():
        ingestor.process_file(input_path, args.platform, output_dir)
    elif input_path.is_dir():
        logger.error("Directory processing not supported in simple version")
        return 1
    else:
        logger.error(f"Input path does not exist: {args.input}")
        return 1

    # Print summary
    print("\nIngestion Summary:")
    print(f"  Documents processed: {ingestor.documents_processed}")
    print(f"  Chunks generated: {ingestor.chunks_generated}")
    print(f"  Duplicates skipped: {ingestor.duplicates_skipped}")
    print(f"  Output directory: {output_dir.absolute()}")

    if ingestor.duplicates_skipped > 0:
        print(f"  Note: {ingestor.duplicates_skipped} duplicate documents were skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
