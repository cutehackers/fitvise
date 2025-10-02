"""PDF processing use case integrating Docling-based processor (Task 1.3.1).

Delivers structured markdown with table extraction using
DoclingPdfProcessor. When optional dependencies aren't available,
processor falls back to pdfminer.six/PyPDF2 and optional Camelot.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.infrastructure.external_services.data_sources.file_processors import (
    DoclingPdfProcessor,
)


def _lib_available(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False


@dataclass
class PdfDocumentResult:
    source: str
    success: bool
    markdown: str
    tables: List[Dict[str, Any]]
    warnings: List[str]
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "success": self.success,
            "markdown": self.markdown,
            "tables": self.tables,
            "warnings": self.warnings,
            "error": self.error,
        }


class ProcessPdfsRequest:
    def __init__(
        self,
        file_paths: Optional[Sequence[str]] = None,
        raw_pdfs: Optional[Sequence[Tuple[bytes, str]]] = None,
        preserve_layout: bool = True,
        extract_tables: bool = True,
    ) -> None:
        self.file_paths = [str(p) for p in (file_paths or [])]
        self.raw_pdfs = list(raw_pdfs or [])
        self.preserve_layout = preserve_layout
        self.extract_tables = extract_tables


class ProcessPdfsResponse:
    def __init__(self, documents: List[PdfDocumentResult], environment: Dict[str, Any]) -> None:
        self.documents = documents
        self.environment = environment

    def as_dict(self) -> Dict[str, Any]:
        return {
            "documents": [d.as_dict() for d in self.documents],
            "environment": self.environment,
        }


class ProcessPdfsUseCase:
    def __init__(self) -> None:
        self._processor = DoclingPdfProcessor()

    async def execute(self, request: ProcessPdfsRequest) -> ProcessPdfsResponse:
        docs: List[PdfDocumentResult] = []

        for path_str in request.file_paths:
            result = await self._process_path(Path(path_str), request)
            docs.append(result)

        for raw, name in request.raw_pdfs:
            result = await self._process_bytes(raw, name, request)
            docs.append(result)

        env = {
            "docling_available": _lib_available("docling") or _lib_available("docling_core"),
            "camelot_available": _lib_available("camelot"),
            "pdfminer_or_pypdf2_available": _lib_available("pdfminer.high_level") or _lib_available("PyPDF2"),
        }
        return ProcessPdfsResponse(documents=docs, environment=env)

    async def _process_path(self, path: Path, request: ProcessPdfsRequest) -> PdfDocumentResult:
        if not path.exists():
            return PdfDocumentResult(
                source=str(path), success=False, markdown="", tables=[], warnings=[], error=f"File not found: {path}"
            )
        result = self._processor.process_pdf_from_path(path)
        return PdfDocumentResult(
            source=str(path),
            success=result.success,
            markdown=result.markdown or result.text,
            tables=result.tables,
            warnings=result.warnings,
            error=result.error,
        )

    async def _process_bytes(self, data: bytes, name: str, request: ProcessPdfsRequest) -> PdfDocumentResult:
        result = self._processor.process_pdf_from_bytes(data, name)
        return PdfDocumentResult(
            source=name,
            success=result.success,
            markdown=result.markdown or result.text,
            tables=result.tables,
            warnings=result.warnings,
            error=result.error,
        )
