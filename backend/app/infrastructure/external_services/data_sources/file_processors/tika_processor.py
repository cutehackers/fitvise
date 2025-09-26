"""Apache Tika integration service with Python fallbacks (Task 1.2.2)."""
from __future__ import annotations

import asyncio
import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:  # Optional import – only used when available at runtime.
    from tika import parser as tika_parser  # type: ignore
    _TIKA_AVAILABLE = True
except Exception:  # pragma: no cover - defensive import guard
    tika_parser = None  # type: ignore
    _TIKA_AVAILABLE = False


@dataclass
class TikaClientConfig:
    """Configuration for talking to a Tika server or library."""

    server_url: Optional[str] = None
    request_timeout_seconds: int = 60
    language_detection: bool = True
    emit_content_type: bool = True
    emit_metadata: bool = True
    max_content_length: Optional[int] = None


@dataclass
class TikaExtractionResult:
    """Structured result returned by the integration service."""

    text: str
    metadata: Dict[str, Any]
    content_type: Optional[str]
    language: Optional[str]
    success: bool
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "metadata": self.metadata,
            "content_type": self.content_type,
            "language": self.language,
            "success": self.success,
            "warnings": self.warnings,
            "error": self.error,
        }


class TikaIntegrationService:
    """Service that extracts text + metadata using Tika with graceful fallbacks."""

    SUPPORTED_SUFFIXES: Tuple[str, ...] = (".pdf", ".doc", ".docx", ".html", ".htm", ".json", ".csv", ".txt", ".md")

    def __init__(self, config: Optional[TikaClientConfig] = None) -> None:
        self.config = config or TikaClientConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def extract_from_path(self, path: Path | str) -> TikaExtractionResult:
        """Extract text & metadata from a file path."""
        file_path = Path(path)
        if not file_path.exists():
            return TikaExtractionResult(
                text="",
                metadata={},
                content_type=None,
                language=None,
                success=False,
                error=f"File not found: {file_path}",
            )

        return await asyncio.to_thread(self._extract_path_sync, file_path)

    async def extract_from_bytes(self, data: bytes, file_name: str = "document") -> TikaExtractionResult:
        """Extract from raw bytes while emulating the provided file name."""
        return await asyncio.to_thread(self._extract_bytes_sync, data, file_name)

    def environment_status(self) -> Dict[str, Any]:
        """Expose diagnostic information about available parsers."""
        return {
            "tika_available": _TIKA_AVAILABLE,
            "supported_suffixes": list(self.SUPPORTED_SUFFIXES),
            "config": {
                "server_url": self.config.server_url,
                "language_detection": self.config.language_detection,
                "emit_metadata": self.config.emit_metadata,
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _extract_path_sync(self, path: Path) -> TikaExtractionResult:
        warnings: List[str] = []
        if _TIKA_AVAILABLE:
            try:
                tika_kwargs = {}
                if self.config.server_url:
                    tika_kwargs["serverEndpoint"] = self.config.server_url
                parsed = tika_parser.from_file(str(path), **tika_kwargs)  # type: ignore[arg-type]
                return self._build_success_result(parsed, warnings)
            except Exception as exc:  # pragma: no cover - we still want fallback
                warnings.append(f"Tika parser failed, falling back to pure-python: {exc}")

        result = self._fallback_extract(path)
        result.warnings.extend(warnings)
        return result

    def _extract_bytes_sync(self, data: bytes, file_name: str) -> TikaExtractionResult:
        warnings: List[str] = []
        if _TIKA_AVAILABLE:
            try:
                tika_kwargs = {}
                if self.config.server_url:
                    tika_kwargs["serverEndpoint"] = self.config.server_url
                parsed = tika_parser.from_buffer(data, **tika_kwargs)
                return self._build_success_result(parsed, warnings)
            except Exception as exc:  # pragma: no cover
                warnings.append(f"Tika parser failed, falling back to pure-python: {exc}")

        suffix = Path(file_name).suffix.lower()
        stream = io.BytesIO(data)
        result = self._fallback_extract_from_stream(stream, suffix)
        result.warnings.extend(warnings)
        return result

    def _build_success_result(self, parsed: Dict[str, Any], warnings: List[str]) -> TikaExtractionResult:
        text = parsed.get("content") or ""
        if self.config.max_content_length is not None:
            text = text[: self.config.max_content_length]
        metadata = parsed.get("metadata") or {}
        content_type = metadata.get("Content-Type") if isinstance(metadata, dict) else None
        language = metadata.get("language") if isinstance(metadata, dict) else None
        return TikaExtractionResult(
            text=text,
            metadata=metadata if isinstance(metadata, dict) else {},
            content_type=content_type,
            language=language,
            success=True,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Pure Python fallbacks
    # ------------------------------------------------------------------
    def _fallback_extract(self, path: Path) -> TikaExtractionResult:
        suffix = path.suffix.lower()
        try:
            with path.open("rb") as file_handle:
                return self._fallback_extract_from_stream(file_handle, suffix, path.name)
        except Exception as exc:
            return TikaExtractionResult(
                text="",
                metadata={},
                content_type=None,
                language=None,
                success=False,
                error=str(exc),
            )

    def _fallback_extract_from_stream(
        self,
        stream: io.BufferedIOBase,
        suffix: str,
        file_name: Optional[str] = None,
    ) -> TikaExtractionResult:
        suffix = suffix or (Path(file_name).suffix.lower() if file_name else "")
        if suffix not in self.SUPPORTED_SUFFIXES:
            return TikaExtractionResult(
                text="",
                metadata={"file_name": file_name, "fallback": True},
                content_type=None,
                language=None,
                success=False,
                error=f"Unsupported file extension for fallback parser: {suffix}",
            )

        handlers = {
            ".pdf": self._parse_pdf,
            ".doc": self._parse_docx,
            ".docx": self._parse_docx,
            ".html": self._parse_html,
            ".htm": self._parse_html,
            ".json": self._parse_json,
            ".csv": self._parse_csv,
            ".txt": self._parse_text,
            ".md": self._parse_text,
        }
        handler = handlers.get(suffix, self._parse_text)
        return handler(stream, file_name)

    def _parse_pdf(self, stream: io.BufferedIOBase, file_name: Optional[str]) -> TikaExtractionResult:
        try:
            from pdfminer.high_level import extract_text  # type: ignore

            stream.seek(0)
            text = extract_text(stream)
            return TikaExtractionResult(
                text=text,
                metadata={"parser": "pdfminer", "file_name": file_name},
                content_type="application/pdf",
                language=self._detect_language(text),
                success=True,
            )
        except Exception:
            try:
                import PyPDF2  # type: ignore

                stream.seek(0)
                reader = PyPDF2.PdfReader(stream)
                text_parts = [page.extract_text() or "" for page in reader.pages]
                text = "\n".join(text_parts)
                return TikaExtractionResult(
                    text=text,
                    metadata={"parser": "pypdf2", "file_name": file_name},
                    content_type="application/pdf",
                    language=self._detect_language(text),
                    success=True,
                )
            except Exception as exc:
                return TikaExtractionResult(
                    text="",
                    metadata={"parser": "none", "file_name": file_name},
                    content_type="application/pdf",
                    language=None,
                    success=False,
                    error=f"PDF parsing failed: {exc}",
                )

    def _parse_docx(self, stream: io.BufferedIOBase, file_name: Optional[str]) -> TikaExtractionResult:
        try:
            import docx  # type: ignore

            stream.seek(0)
            document = docx.Document(stream)
            text = "\n".join(paragraph.text for paragraph in document.paragraphs)
            return TikaExtractionResult(
                text=text,
                metadata={"parser": "python-docx", "file_name": file_name},
                content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                language=self._detect_language(text),
                success=True,
            )
        except Exception as exc:
            return TikaExtractionResult(
                text="",
                metadata={"parser": "none", "file_name": file_name},
                content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                language=None,
                success=False,
                error=f"DOC/DOCX parsing failed: {exc}",
            )

    def _parse_html(self, stream: io.BufferedIOBase, file_name: Optional[str]) -> TikaExtractionResult:
        text = ""
        try:
            from bs4 import BeautifulSoup  # type: ignore

            stream.seek(0)
            soup = BeautifulSoup(stream.read(), "html.parser")
            text = soup.get_text(separator="\n")
        except Exception:
            stream.seek(0)
            text = stream.read().decode("utf-8", errors="ignore")
        return TikaExtractionResult(
            text=text,
            metadata={"parser": "beautifulsoup", "file_name": file_name},
            content_type="text/html",
            language=self._detect_language(text),
            success=True,
        )

    def _parse_json(self, stream: io.BufferedIOBase, file_name: Optional[str]) -> TikaExtractionResult:
        stream.seek(0)
        try:
            data = json.load(io.TextIOWrapper(stream, encoding="utf-8"))
            text = json.dumps(data, indent=2)
            return TikaExtractionResult(
                text=text,
                metadata={"parser": "json", "file_name": file_name},
                content_type="application/json",
                language=None,
                success=True,
            )
        except Exception as exc:
            return TikaExtractionResult(
                text="",
                metadata={"parser": "json", "file_name": file_name},
                content_type="application/json",
                language=None,
                success=False,
                error=f"JSON parsing failed: {exc}",
            )

    def _parse_csv(self, stream: io.BufferedIOBase, file_name: Optional[str]) -> TikaExtractionResult:
        stream.seek(0)
        try:
            text_stream = io.TextIOWrapper(stream, encoding="utf-8")
            reader = csv.reader(text_stream)
            rows = [", ".join(row) for row in reader]
            text = "\n".join(rows)
            return TikaExtractionResult(
                text=text,
                metadata={"parser": "csv", "file_name": file_name},
                content_type="text/csv",
                language=None,
                success=True,
            )
        except Exception as exc:
            return TikaExtractionResult(
                text="",
                metadata={"parser": "csv", "file_name": file_name},
                content_type="text/csv",
                language=None,
                success=False,
                error=f"CSV parsing failed: {exc}",
            )

    def _parse_text(self, stream: io.BufferedIOBase, file_name: Optional[str]) -> TikaExtractionResult:
        stream.seek(0)
        text = stream.read().decode("utf-8", errors="ignore")
        return TikaExtractionResult(
            text=text,
            metadata={"parser": "text", "file_name": file_name},
            content_type="text/plain",
            language=self._detect_language(text),
            success=True,
        )

    # ------------------------------------------------------------------
    def _detect_language(self, text: str) -> Optional[str]:
        if not text or not self.config.language_detection:
            return None
        try:
            from langdetect import detect  # type: ignore

            return detect(text)
        except Exception:
            return None
