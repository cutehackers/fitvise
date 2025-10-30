#!/usr/bin/env python3
"""
RAG Infrastructure Setup and Validation Script

Phase 1 of RAG build pipeline. Validates all required services and configurations
before proceeding with document ingestion and embedding generation.

Usage:
    python scripts/setup_rag_infrastructure.py --config rag_pipeline.yaml [--verbose]
"""

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add backend/ to sys.path so "app" imports resolve when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline.config import PipelineSpec
from app.application.use_cases.embedding.setup_embedding_infrastructure import (
    SetupEmbeddingInfrastructureUseCase,
    SetupRequest,
    SetupResponse,
)
from app.infrastructure.external_services.vector_stores.weaviate_schema import WeaviateSchema
from app.infrastructure.storage.object_storage.minio_client import ObjectStorageClient, ObjectStorageConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of infrastructure validation."""

    success: bool
    validation_results: Dict[str, Any]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "validation_results": self.validation_results,
            "errors": self.errors,
            "warnings": self.warnings,
            "total_components": len(self.validation_results),
            "failed_components": len([e for e in self.errors if "Critical:" in e]),
        }


class InfrastructureValidationError(Exception):
    """Critical infrastructure validation failed."""

    def __init__(self, message: str, errors: List[str], validation_results: Dict[str, Any]):
        super().__init__(message)
        self.errors = errors
        self.validation_results = validation_results


def _storage_client(spec: PipelineSpec) -> ObjectStorageClient:
    """Instantiate an object storage client using pipeline configuration."""
    return ObjectStorageClient(
        ObjectStorageConfig(
            provider=spec.storage.provider,
            endpoint=spec.storage.endpoint,
            access_key=spec.storage.access_key,
            secret_key=spec.storage.secret_key,
            secure=spec.storage.secure,
            base_dir=spec.storage.base_dir,
        )
    )


async def validate_infrastructure(config: PipelineSpec, verbose: bool = False) -> ValidationResult:
    """Validate all RAG infrastructure components.

    Args:
        config: Pipeline specification
        verbose: Whether to log detailed validation information

    Returns:
        ValidationResult with comprehensive status information

    Raises:
        InfrastructureValidationError: If critical validation fails
    """
    validation_results = {}
    errors = []

    logger.info("Starting RAG infrastructure validation...")

    # 1. Embedding Service Validation (uses existing SetupEmbeddingInfrastructureUseCase)
    logger.info("Validating embedding service...")
    try:
        setup_use_case = SetupEmbeddingInfrastructureUseCase()

        # Prepare embedding configuration
        embedding_config = {
            "model_name": "Alibaba-NLP/gte-multilingual-base",
            "device": "auto",  # Let the service decide
            "trust_remote_code": True
        }

        # Prepare Weaviate configuration
        weaviate_config = {
            "url": "http://localhost:8080",  # Default Weaviate URL
            "timeout_config": (10, 30)  # Connection and request timeouts
        }

        setup_request = SetupRequest(
            vector_dimension=768,
            embedding_config=embedding_config,
            weaviate_config=weaviate_config,
            recreate_schema=False  # Don't recreate existing schema
        )

        embedding_response = await setup_use_case.execute(setup_request)
        validation_results["embedding_service"] = embedding_response.as_dict()

        if verbose:
            logger.info(f"Embedding service status: {embedding_response.as_dict()}")

        if not embedding_response.success:
            errors.extend([f"Critical: {error}" for error in embedding_response.errors])
        else:
            logger.info("✅ Embedding service validation successful")

    except Exception as e:
        error_msg = f"Critical: Embedding service validation failed: {str(e)}"
        errors.append(error_msg)
        validation_results["embedding_service"] = {"error": str(e)}
        logger.error(error_msg)

    # 2. Weaviate Schema Validation
    logger.info("Validating Weaviate schema...")
    try:
        # Use the setup use case to get the Weaviate client
        if 'embedding_service' in validation_results and validation_results["embedding_service"].get("weaviate", {}).get("connected"):
            # If embedding setup succeeded, we should have a connected client
            setup_use_case = SetupEmbeddingInfrastructureUseCase()
            await setup_use_case.execute(SetupRequest())  # Re-establish connection
            weaviate_client = setup_use_case.get_weaviate_client()

            if weaviate_client and weaviate_client.is_connected:
                # Verify DocumentChunk schema exists
                schema = WeaviateSchema(weaviate_client._client)
                chunk_schema_exists = await schema.class_exists("Chunk")
                document_schema_exists = await schema.class_exists("DocumentChunk")

                # Check if either schema exists (DocumentChunk is preferred)
                schema_exists = document_schema_exists or chunk_schema_exists
                schema_name = "DocumentChunk" if document_schema_exists else "Chunk"

                validation_results["weaviate_schema"] = {
                    "connected": True,
                    "schema_exists": schema_exists,
                    "schema_name": schema_name,
                    "url": weaviate_client.config.get_url(),
                    "vector_dimension": 768
                }

                if not schema_exists:
                    errors.append("Critical: DocumentChunk or Chunk schema not found in Weaviate")
                    logger.error("❌ No valid schema found in Weaviate")
                else:
                    logger.info(f"✅ Weaviate schema validation successful (found {schema_name})")

                if verbose:
                    logger.info(f"Weaviate connection: {weaviate_client.config.get_url()}")
            else:
                errors.append("Critical: Weaviate client connection failed")
                logger.error("❌ Weaviate client not connected")
        else:
            errors.append("Critical: Weaviate connection failed during embedding setup")
            logger.error("❌ Weaviate not connected from embedding setup")

    except Exception as e:
        error_msg = f"Critical: Weaviate validation failed: {str(e)}"
        errors.append(error_msg)
        validation_results["weaviate_schema"] = {"error": str(e)}
        logger.error(error_msg)

    # 3. Object Storage Validation
    logger.info("Validating object storage...")
    try:
        storage_client = _storage_client(config)

        # Test bucket access - try to list buckets or create if not exists
        try:
            buckets = await storage_client.list_buckets()
            bucket_names = [bucket.name for bucket in buckets]
            bucket_accessible = config.storage.bucket in bucket_names

            if not bucket_accessible:
                # Try to create the bucket
                logger.info(f"Bucket '{config.storage.bucket}' not found, attempting to create...")
                await storage_client.create_bucket(config.storage.bucket)
                bucket_accessible = True
                logger.info(f"✅ Created bucket '{config.storage.bucket}'")
            else:
                logger.info(f"✅ Bucket '{config.storage.bucket}' already exists")

            # Test write/read operation with a small test object
            test_key = f"validation-test/{hash('test')}.txt"
            test_content = b"RAG infrastructure validation test object"

            # Write test object
            await storage_client.put_object(
                bucket_name=config.storage.bucket,
                object_key=test_key,
                data=test_content,
                content_type="text/plain"
            )

            # Read test object back
            result = await storage_client.get_object(config.storage.bucket, test_key)
            read_success = result.data == test_content

            # Clean up test object
            await storage_client.delete_object(config.storage.bucket, test_key)

            validation_results["object_storage"] = {
                "provider": config.storage.provider,
                "bucket_accessible": bucket_accessible,
                "bucket_name": config.storage.bucket,
                "read_write_test": read_success,
                "endpoint": config.storage.endpoint
            }

            if not read_success:
                errors.append("Critical: Object storage read/write test failed")
                logger.error("❌ Object storage read/write test failed")
            else:
                logger.info("✅ Object storage validation successful")

        except Exception as storage_error:
            error_msg = f"Critical: Object storage operations failed: {str(storage_error)}"
            errors.append(error_msg)
            validation_results["object_storage"] = {"error": str(storage_error)}
            logger.error(error_msg)

    except Exception as e:
        error_msg = f"Critical: Object storage validation failed: {str(e)}"
        errors.append(error_msg)
        validation_results["object_storage"] = {"error": str(e)}
        logger.error(error_msg)

    # 4. Configuration Validation
    logger.info("Validating pipeline configuration...")
    try:
        config_errors = []

        # Validate document path
        if not config.documents.path.exists():
            config_errors.append(f"Document path does not exist: {config.documents.path}")
        elif not config.documents.path.is_dir():
            config_errors.append(f"Document path is not a directory: {config.documents.path}")

        # Validate document patterns
        if not config.documents.include:
            config_errors.append("No document patterns specified (include field is empty)")

        # Validate storage configuration
        if not config.storage.bucket:
            config_errors.append("Storage bucket name not specified")

        # Count potential documents
        if config.documents.path.exists():
            file_count = 0
            for pattern in config.documents.include:
                files = list(config.documents.path.glob(pattern)) if not config.documents.recurse else list(config.documents.path.rglob(pattern))
                file_count += len([f for f in files if f.is_file()])

            validation_results["configuration"] = {
                "valid": len(config_errors) == 0,
                "document_path": str(config.documents.path),
                "patterns": config.documents.include,
                "recurse": config.documents.recurse,
                "estimated_files": file_count,
                "storage_bucket": config.storage.bucket,
                "storage_provider": config.storage.provider
            }

            if file_count == 0:
                config_errors.append("No files found matching document patterns")
                logger.warning("⚠️  No files found with specified patterns")
            else:
                logger.info(f"✅ Found {file_count} files matching patterns")
        else:
            validation_results["configuration"] = {
                "valid": len(config_errors) == 0,
                "document_path": str(config.documents.path),
                "patterns": config.documents.include,
                "recurse": config.documents.recurse,
                "estimated_files": 0,
                "storage_bucket": config.storage.bucket,
                "storage_provider": config.storage.provider
            }

        errors.extend(config_errors)

        if len(config_errors) == 0:
            logger.info("✅ Configuration validation successful")
        else:
            for error in config_errors:
                logger.warning(f"⚠️  Configuration issue: {error}")

    except Exception as e:
        error_msg = f"Configuration validation failed: {str(e)}"
        errors.append(error_msg)
        validation_results["configuration"] = {"error": str(e)}
        logger.error(error_msg)

    # Categorize errors into critical and warnings
    critical_errors = [e for e in errors if "Critical:" in e]
    warnings = [e for e in errors if "Critical:" not in e]

    # Fail Fast: If any critical errors, raise exception
    if critical_errors:
        logger.error(f"❌ Infrastructure validation failed with {len(critical_errors)} critical errors")
        for error in critical_errors:
            logger.error(f"   - {error}")

        raise InfrastructureValidationError(
            message="Critical infrastructure validation failed",
            errors=critical_errors,
            validation_results=validation_results
        )

    # Success case
    logger.info("🎉 RAG infrastructure validation completed successfully!")
    if warnings:
        logger.warning(f"⚠️  {len(warnings)} warnings found:")
        for warning in warnings:
            logger.warning(f"   - {warning}")

    return ValidationResult(
        success=len(critical_errors) == 0,
        validation_results=validation_results,
        errors=errors,
        warnings=warnings
    )


async def main() -> int:
    """Main function to run infrastructure validation."""
    parser = argparse.ArgumentParser(description="Validate RAG infrastructure setup")
    parser.add_argument("--config", required=True, help="Path to rag_pipeline.yaml")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--output", help="Path to save validation results as JSON")

    args = parser.parse_args()

    try:
        # Load pipeline configuration
        spec = PipelineSpec.from_file(args.config)

        # Run validation
        result = await validate_infrastructure(spec, verbose=args.verbose)

        # Save results if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(result.as_dict(), f, indent=2)
            logger.info(f"Validation results saved to {args.output}")

        # Print summary
        print("\n" + "="*60)
        print("RAG INFRASTRUCTURE VALIDATION SUMMARY")
        print("="*60)
        print(f"Success: {'✅ YES' if result.success else '❌ NO'}")
        print(f"Components Validated: {result.as_dict()['total_components']}")
        print(f"Failed Components: {result.as_dict()['failed_components']}")
        print(f"Warnings: {len(result.warnings)}")

        if result.warnings:
            print("\nWarnings:")
            for warning in result.warnings:
                print(f"  ⚠️  {warning}")

        if not result.success:
            print("\nCritical Errors:")
            for error in result.errors:
                if "Critical:" in error:
                    print(f"  ❌ {error}")
            return 1
        else:
            print("\n🎉 All critical components validated successfully!")
            return 0

    except Exception as e:
        logger.error(f"Infrastructure validation failed: {str(e)}")
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))