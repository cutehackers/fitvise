#!/usr/bin/env python3
"""
Simple Ingestion Script for Testing

Simplified version of the Docling ingestion script that focuses on core functionality
and uses only local storage for testing purposes.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add botadvisor to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from app.core.entity.document import Document
from app.core.entity.document_metadata import DocumentMetadata
from app.core.entity.chunk import Chunk
from app.observability.langfuse import get_tracer
from app.observability.logging import get_logger, configure_logger
from app.storage.local_storage import LocalStorage

# Configure logging
configure_logger()
logger = get_logger("ingest")
tracer = get_tracer()

# Dynamic Docling import with fallback
DOCLING_AVAILABLE = False
PDFReader = None
OfficeReader = None
TextReader = None

try:
    import docling
    from docling.readers import PDFReader, OfficeReader, TextReader

    DOCLING_AVAILABLE = True
    logger.info("Docling imported successfully")
except ImportError as e:
    logger.warning(f"Docling not available: {e}. Using fallback text processing.")
    DOCLING_AVAILABLE = False


class SimpleIngestionScript:
    """
    Simplified document ingestion script for testing.

    Focuses on core functionality with local storage only.
    """

    def __init__(self):
        """Initialize ingestion script with local storage backend."""
        self.storage_backend = self._initialize_storage_backend()
        self.duplicates_skipped = 0
        self.documents_processed = 0
        self.chunks_generated = 0

    def _initialize_storage_backend(self):
        """Initialize local storage backend."""
        base_path = os.environ.get("STORAGE_LOCAL_PATH", "./data/artifacts")
        return LocalStorage(base_path=base_path)

    def _get_reader_for_file(self, file_path: Path) -> Optional[Any]:
        """
        Get appropriate document reader for file based on extension.

        Args:
            file_path: Path to file

        Returns:
            Reader function or None if unsupported
        """
        extension = file_path.suffix.lower()

        if not DOCLING_AVAILABLE:
            # Fallback: simple text reading for testing
            if extension in (".txt", ".md", ".html", ".htm"):
                return self._simple_text_reader
            else:
                logger.warning("Docling not available, only text files supported", extra={"file": str(file_path)})
                return None

        # Use Docling readers when available
        if extension == ".pdf":
            return PDFReader()
        elif extension in (".doc", ".docx"):
            return OfficeReader()
        elif extension in (".txt", ".md", ".html", ".htm"):
            return TextReader()
        else:
            logger.warning(f"Unsupported file type: {extension}", extra={"file": str(file_path)})
            return None

    def _simple_text_reader(self, content: bytes) -> str:
        """
        Simple text reader fallback when Docling is not available.

        Args:
            content: Raw content bytes

        Returns:
            Decoded text content
        """
        try:
            # Try UTF-8 first
            return content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                # Fallback to latin-1 which can decode any byte sequence
                return content.decode("latin-1")
            except Exception:
                # Last resort: decode with errors replaced
                return content.decode("utf-8", errors="replace")

    def _extract_metadata_from_path(self, file_path: Path, platform: str) -> Dict[str, str]:
        """
        Extract metadata from file path and platform.

        Args:
            file_path: Path to source file
            platform: Source platform

        Returns:
            Dictionary of extracted metadata
        """
        return {
            "source_id": str(file_path),
            "platform": platform,
            "source_url": f"file://{file_path.absolute()}",
            "file_name": file_path.name,
            "file_extension": file_path.suffix,
            "file_size": str(file_path.stat().st_size),
        }

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
        metadata = self._extract_metadata_from_path(file_path, platform)

        return Document.create(
            source_id=metadata["source_id"],
            platform=platform,
            source_url=metadata["source_url"],
            content=content,
            mime_type=self._get_mime_type(file_path),
        )

    def _get_mime_type(self, file_path: Path) -> str:
        """
        Get MIME type for file based on extension.

        Args:
            file_path: Path to file

        Returns:
            MIME type string
        """
        extension = file_path.suffix.lower()
        mime_types = {
            ".pdf": "application/pdf",
            ".doc": "application/msword",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".html": "text/html",
            ".htm": "text/html",
            ".json": "application/json",
            ".xml": "application/xml",
            ".xls": "application/vnd.ms-excel",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".ppt": "application/vnd.ms-powerpoint",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
        }
        return mime_types.get(extension, "application/octet-stream")

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
            reader = self._get_reader_for_file(file_path)
            if not reader:
                return None

            # Parse document content
            if hasattr(reader, "read"):
                raw_text = reader.read(content)
            else:
                # Fallback for simple reader functions
                raw_text = reader(content)

            return document, raw_text

        except Exception as e:
            logger.error(
                f"Failed to parse document {file_path.name}: {str(e)}", extra={"file": str(file_path), "error": str(e)}
            )
            return None

    def _chunk_text(self, text: str, document: Document) -> List[Chunk]:
        """
        Chunk text into manageable pieces with metadata.

        Args:
            text: Raw text content
            document: Document entity

        Returns:
            List of Chunk entities
        """
        # Simple chunking strategy - can be enhanced later
        chunk_size = 1000  # characters per chunk
        chunks = []

        # Split text into chunks
        for i, start in enumerate(range(0, len(text), chunk_size)):
            end = start + chunk_size
            chunk_content = text[start:end]

            # Create chunk with metadata
            chunk = Chunk(
                chunk_id=f"{document.id}_chunk_{i}",
                content=chunk_content.strip(),
                metadata=DocumentMetadata(
                    doc_id=document.id,
                    source_id=document.source_id,
                    platform=document.platform,
                    source_url=document.source_url,
                    page=None,  # Could be extracted from PDF
                    section=f"chunk_{i}",
                ),
            )
            chunks.append(chunk)

        self.chunks_generated += len(chunks)
        return chunks

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
            chunks = self._chunk_text(raw_text, document)

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


def main():
    """Main entry point for ingestion script."""
    parser = argparse.ArgumentParser(
        description="BotAdvisor Simple Document Ingestion Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--input", "-i", required=True, help="Input file or directory path")

    parser.add_argument("--out", "-o", default="./data/chunks", help="Output directory for processed chunks")

    parser.add_argument("--platform", "-p", required=True, help="Source platform (e.g., filesystem, web, gdrive)")

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        os.environ["LOG_LEVEL"] = "DEBUG"
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
        sys.exit(1)
    else:
        logger.error(f"Input path does not exist: {args.input}")
        sys.exit(1)

    # Print summary
    print("\nIngestion Summary:")
    print(f"  Documents processed: {ingestor.documents_processed}")
    print(f"  Chunks generated: {ingestor.chunks_generated}")
    print(f"  Duplicates skipped: {ingestor.duplicates_skipped}")
    print(f"  Output directory: {output_dir.absolute()}")

    if ingestor.duplicates_skipped > 0:
        print(f"  Note: {ingestor.duplicates_skipped} duplicate documents were skipped")


if __name__ == "__main__":
    main()
