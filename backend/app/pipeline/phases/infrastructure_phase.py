"""Infrastructure Validation Phase.

This phase validates all required infrastructure components before proceeding
with document ingestion and embedding generation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.pipeline.config import PipelineSpec
from app.application.use_cases.embedding.setup_embedding_infrastructure import (
    SetupEmbeddingInfrastructureUseCase,
    SetupRequest,
)
from app.infrastructure.external_services.vector_stores.weaviate_schema import WeaviateSchema
from app.infrastructure.storage.object_storage.minio_client import (
    ObjectStorageClient,
    ObjectStorageConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class InfrastructureValidationResult:
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

    def __init__(
        self, message: str, errors: List[str], validation_results: Dict[str, Any]
    ):
        super().__init__(message)
        self.errors = errors
        self.validation_results = validation_results


class InfrastructurePhase:
    """Phase 1: Infrastructure Setup and Validation.

    Validates all required services and configurations before proceeding
    with document ingestion and embedding generation.
    """

    def __init__(self, verbose: bool = False):
        """Initialize the infrastructure phase.

        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

    def _create_storage_client(self, spec: PipelineSpec) -> ObjectStorageClient:
        """Create an object storage client from pipeline specification.

        Args:
            spec: Pipeline specification

        Returns:
            Configured object storage client
        """
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

    async def _validate_embedding_service(
        self, validation_results: Dict[str, Any], errors: List[str]
    ) -> None:
        """Validate embedding service setup.

        Args:
            validation_results: Dictionary to store validation results
            errors: List to append errors to
        """
        logger.info("Validating embedding service...")
        try:
            setup_use_case = SetupEmbeddingInfrastructureUseCase()

            # Prepare embedding configuration
            embedding_config = {
                "model_name": "Alibaba-NLP/gte-multilingual-base",
                "device": "auto",
                "trust_remote_code": True,
            }

            # Prepare Weaviate configuration
            weaviate_config = {
                "url": "http://localhost:8080",
                "timeout_config": (10, 30),
            }

            setup_request = SetupRequest(
                vector_dimension=768,
                embedding_config=embedding_config,
                weaviate_config=weaviate_config,
                recreate_schema=False,
            )

            embedding_response = await setup_use_case.execute(setup_request)
            validation_results["embedding_service"] = embedding_response.as_dict()

            if self.verbose:
                logger.info(f"Embedding service status: {embedding_response.as_dict()}")

            if not embedding_response.success:
                errors.extend(
                    [f"Critical: {error}" for error in embedding_response.errors]
                )
            else:
                logger.info("✅ Embedding service validation successful")

        except Exception as e:
            error_msg = f"Critical: Embedding service validation failed: {str(e)}"
            errors.append(error_msg)
            validation_results["embedding_service"] = {"error": str(e)}
            logger.error(error_msg)

    async def _validate_weaviate_schema(
        self, validation_results: Dict[str, Any], errors: List[str]
    ) -> None:
        """Validate Weaviate schema exists.

        Args:
            validation_results: Dictionary to store validation results
            errors: List to append errors to
        """
        logger.info("Validating Weaviate schema...")
        try:
            if "embedding_service" in validation_results and validation_results[
                "embedding_service"
            ].get("weaviate", {}).get("connected"):
                setup_use_case = SetupEmbeddingInfrastructureUseCase()
                await setup_use_case.execute(SetupRequest())
                weaviate_client = setup_use_case.get_weaviate_client()

                if weaviate_client and weaviate_client.is_connected:
                    schema = WeaviateSchema(weaviate_client._client)
                    chunk_schema_exists = await schema.class_exists("Chunk")
                    document_schema_exists = await schema.class_exists("DocumentChunk")

                    schema_exists = document_schema_exists or chunk_schema_exists
                    schema_name = (
                        "DocumentChunk" if document_schema_exists else "Chunk"
                    )

                    validation_results["weaviate_schema"] = {
                        "connected": True,
                        "schema_exists": schema_exists,
                        "schema_name": schema_name,
                        "url": weaviate_client.config.get_url(),
                        "vector_dimension": 768,
                    }

                    if not schema_exists:
                        errors.append(
                            "Critical: DocumentChunk or Chunk schema not found in Weaviate"
                        )
                        logger.error("❌ No valid schema found in Weaviate")
                    else:
                        logger.info(
                            f"✅ Weaviate schema validation successful (found {schema_name})"
                        )

                    if self.verbose:
                        logger.info(
                            f"Weaviate connection: {weaviate_client.config.get_url()}"
                        )
                else:
                    errors.append("Critical: Weaviate client connection failed")
                    logger.error("❌ Weaviate client not connected")
            else:
                errors.append(
                    "Critical: Weaviate connection failed during embedding setup"
                )
                logger.error("❌ Weaviate not connected from embedding setup")

        except Exception as e:
            error_msg = f"Critical: Weaviate validation failed: {str(e)}"
            errors.append(error_msg)
            validation_results["weaviate_schema"] = {"error": str(e)}
            logger.error(error_msg)

    async def _validate_object_storage(
        self, spec: PipelineSpec, validation_results: Dict[str, Any], errors: List[str]
    ) -> None:
        """Validate object storage accessibility.

        Args:
            spec: Pipeline specification
            validation_results: Dictionary to store validation results
            errors: List to append errors to
        """
        logger.info("Validating object storage...")
        try:
            storage_client = self._create_storage_client(spec)

            try:
                buckets = await storage_client.list_buckets()
                bucket_names = [bucket.name for bucket in buckets]
                bucket_accessible = spec.storage.bucket in bucket_names

                if not bucket_accessible:
                    logger.info(
                        f"Bucket '{spec.storage.bucket}' not found, attempting to create..."
                    )
                    await storage_client.create_bucket(spec.storage.bucket)
                    bucket_accessible = True
                    logger.info(f"✅ Created bucket '{spec.storage.bucket}'")
                else:
                    logger.info(f"✅ Bucket '{spec.storage.bucket}' already exists")

                # Test write/read operation
                test_key = f"validation-test/{hash('test')}.txt"
                test_content = b"RAG infrastructure validation test object"

                await storage_client.put_object(
                    bucket_name=spec.storage.bucket,
                    object_key=test_key,
                    data=test_content,
                    content_type="text/plain",
                )

                result = await storage_client.get_object(spec.storage.bucket, test_key)
                read_success = result.data == test_content

                await storage_client.delete_object(spec.storage.bucket, test_key)

                validation_results["object_storage"] = {
                    "provider": spec.storage.provider,
                    "bucket_accessible": bucket_accessible,
                    "bucket_name": spec.storage.bucket,
                    "read_write_test": read_success,
                    "endpoint": spec.storage.endpoint,
                }

                if not read_success:
                    errors.append("Critical: Object storage read/write test failed")
                    logger.error("❌ Object storage read/write test failed")
                else:
                    logger.info("✅ Object storage validation successful")

            except Exception as storage_error:
                error_msg = (
                    f"Critical: Object storage operations failed: {str(storage_error)}"
                )
                errors.append(error_msg)
                validation_results["object_storage"] = {"error": str(storage_error)}
                logger.error(error_msg)

        except Exception as e:
            error_msg = f"Critical: Object storage validation failed: {str(e)}"
            errors.append(error_msg)
            validation_results["object_storage"] = {"error": str(e)}
            logger.error(error_msg)

    async def _validate_configuration(
        self, spec: PipelineSpec, validation_results: Dict[str, Any], errors: List[str]
    ) -> None:
        """Validate pipeline configuration.

        Args:
            spec: Pipeline specification
            validation_results: Dictionary to store validation results
            errors: List to append errors to
        """
        logger.info("Validating pipeline configuration...")
        try:
            config_errors = []

            # Validate document path
            if not spec.documents.path.exists():
                config_errors.append(
                    f"Document path does not exist: {spec.documents.path}"
                )
            elif not spec.documents.path.is_dir():
                config_errors.append(
                    f"Document path is not a directory: {spec.documents.path}"
                )

            # Validate document patterns
            if not spec.documents.include:
                config_errors.append(
                    "No document patterns specified (include field is empty)"
                )

            # Validate storage configuration
            if not spec.storage.bucket:
                config_errors.append("Storage bucket name not specified")

            # Count potential documents
            if spec.documents.path.exists():
                file_count = 0
                for pattern in spec.documents.include:
                    files = (
                        list(spec.documents.path.glob(pattern))
                        if not spec.documents.recurse
                        else list(spec.documents.path.rglob(pattern))
                    )
                    file_count += len([f for f in files if f.is_file()])

                validation_results["configuration"] = {
                    "valid": len(config_errors) == 0,
                    "document_path": str(spec.documents.path),
                    "patterns": spec.documents.include,
                    "recurse": spec.documents.recurse,
                    "estimated_files": file_count,
                    "storage_bucket": spec.storage.bucket,
                    "storage_provider": spec.storage.provider,
                }

                if file_count == 0:
                    config_errors.append("No files found matching document patterns")
                    logger.warning("⚠️  No files found with specified patterns")
                else:
                    logger.info(f"✅ Found {file_count} files matching patterns")
            else:
                validation_results["configuration"] = {
                    "valid": len(config_errors) == 0,
                    "document_path": str(spec.documents.path),
                    "patterns": spec.documents.include,
                    "recurse": spec.documents.recurse,
                    "estimated_files": 0,
                    "storage_bucket": spec.storage.bucket,
                    "storage_provider": spec.storage.provider,
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

    async def execute(
        self, spec: PipelineSpec
    ) -> InfrastructureValidationResult:
        """Execute infrastructure validation phase.

        Args:
            spec: Pipeline specification

        Returns:
            InfrastructureValidationResult with comprehensive status

        Raises:
            InfrastructureValidationError: If critical validation fails
        """
        validation_results = {}
        errors = []

        logger.info("Starting RAG infrastructure validation...")

        # Run all validations
        await self._validate_embedding_service(validation_results, errors)
        await self._validate_weaviate_schema(validation_results, errors)
        await self._validate_object_storage(spec, validation_results, errors)
        await self._validate_configuration(spec, validation_results, errors)

        # Categorize errors
        critical_errors = [e for e in errors if "Critical:" in e]
        warnings = [e for e in errors if "Critical:" not in e]

        # Fail fast on critical errors
        if critical_errors:
            logger.error(
                f"❌ Infrastructure validation failed with {len(critical_errors)} critical errors"
            )
            for error in critical_errors:
                logger.error(f"   - {error}")

            raise InfrastructureValidationError(
                message="Critical infrastructure validation failed",
                errors=critical_errors,
                validation_results=validation_results,
            )

        # Success case
        logger.info("🎉 RAG infrastructure validation completed successfully!")
        if warnings:
            logger.warning(f"⚠️  {len(warnings)} warnings found:")
            for warning in warnings:
                logger.warning(f"   - {warning}")

        return InfrastructureValidationResult(
            success=len(critical_errors) == 0,
            validation_results=validation_results,
            errors=errors,
            warnings=warnings,
        )
