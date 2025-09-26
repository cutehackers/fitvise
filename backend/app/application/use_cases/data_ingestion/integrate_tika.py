"""Use case orchestrating Apache Tika integration (Task 1.2.2)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from app.infrastructure.external_services.data_sources.file_processors import (
    TikaClientConfig,
    TikaIntegrationService,
    TikaExtractionResult,
)


@dataclass
class TikaDocumentResult:
    """Container for per-document extraction details."""

    source: str
    result: TikaExtractionResult

    def as_dict(self) -> Dict[str, object]:
        payload = self.result.as_dict()
        payload["source"] = self.source
        return payload


class IntegrateTikaRequest:
    """Request for running text extraction via Tika."""

    def __init__(
        self,
        file_paths: Optional[Sequence[str]] = None,
        raw_documents: Optional[Sequence[Tuple[bytes, str]]] = None,
        config: Optional[TikaClientConfig] = None,
    ) -> None:
        self.file_paths = [str(path) for path in (file_paths or [])]
        self.raw_documents = list(raw_documents or [])
        self.config = config


class IntegrateTikaResponse:
    """Response summarising extraction outcomes."""

    def __init__(
        self,
        documents: List[TikaDocumentResult],
        environment_status: Dict[str, object],
    ) -> None:
        self.documents = documents
        self.environment_status = environment_status

    @property
    def success_count(self) -> int:
        return len([doc for doc in self.documents if doc.result.success])

    @property
    def failure_count(self) -> int:
        return len([doc for doc in self.documents if not doc.result.success])

    def as_dict(self) -> Dict[str, object]:
        return {
            "documents": [doc.as_dict() for doc in self.documents],
            "environment_status": self.environment_status,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
        }


class IntegrateTikaUseCase:
    """Runs Tika extraction for provided files and byte streams."""

    def __init__(self, tika_service: Optional[TikaIntegrationService] = None) -> None:
        self.tika_service = tika_service or TikaIntegrationService()

    async def execute(self, request: IntegrateTikaRequest) -> IntegrateTikaResponse:
        if request.config:
            self.tika_service = TikaIntegrationService(config=request.config)
        results: List[TikaDocumentResult] = []

        for path_str in request.file_paths:
            extraction = await self.tika_service.extract_from_path(Path(path_str))
            results.append(TikaDocumentResult(source=path_str, result=extraction))

        for raw_bytes, file_name in request.raw_documents:
            extraction = await self.tika_service.extract_from_bytes(raw_bytes, file_name)
            results.append(TikaDocumentResult(source=file_name, result=extraction))

        status = self.tika_service.environment_status()
        return IntegrateTikaResponse(documents=results, environment_status=status)
