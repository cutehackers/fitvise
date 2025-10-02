"""Docling-based PDF processor with robust fallbacks (Task 1.3.1).

Attempts to use Docling for structured markdown conversion. Falls back to
pdfminer.six/PyPDF2 for text and Camelot for table extraction when available.
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .base_processor import FileProcessingResult, PdfProcessorBase


def _try_import_docling():
    try:
        import docling  # type: ignore

        return docling
    except Exception:
        try:
            from docling_core import document_converter  # type: ignore

            return document_converter
        except Exception:
            return None


def _extract_text_pdfminer(stream: io.BytesIO) -> str:
    try:
        from pdfminer.high_level import extract_text  # type: ignore

        stream.seek(0)
        return extract_text(stream)
    except Exception:
        try:
            import PyPDF2  # type: ignore

            stream.seek(0)
            reader = PyPDF2.PdfReader(stream)
            parts = [(page.extract_text() or "") for page in reader.pages]
            return "\n".join(parts)
        except Exception:
            return ""


def _extract_tables_camelot(path: Path) -> List[pd.DataFrame]:
    try:
        import camelot  # type: ignore

        tables = camelot.read_pdf(str(path), pages="all")
        dfs: List[pd.DataFrame] = []
        for t in tables:
            try:
                dfs.append(t.df)
            except Exception:
                continue
        return dfs
    except Exception:
        return []


class DoclingPdfProcessor(PdfProcessorBase):
    SUPPORTED_SUFFIXES = (".pdf",)

    def __init__(self, preserve_layout: bool = True, extract_tables: bool = True) -> None:
        self._docling = _try_import_docling()
        self.preserve_layout = preserve_layout
        self.extract_tables = extract_tables

    def process_pdf_from_path(self, path: Path) -> FileProcessingResult:
        if not path.exists():
            return FileProcessingResult(success=False, error=f"File not found: {path}")
        if self._docling is not None:
            try:
                markdown, tables, warnings = self._docling_convert_path(path)
                return FileProcessingResult(
                    success=True,
                    text="",
                    markdown=markdown,
                    tables=tables,
                    metadata={"processor": "docling", "file_name": path.name},
                    content_type="application/pdf",
                    warnings=warnings,
                )
            except Exception as exc:
                return self._fallback_path(path, warnings=[f"Docling failed: {exc}"])
        return self._fallback_path(path)

    def process_pdf_from_bytes(self, data: bytes, file_name: str = "document.pdf") -> FileProcessingResult:
        if self._docling is not None:
            try:
                markdown, tables, warnings = self._docling_convert_bytes(data)
                return FileProcessingResult(
                    success=True,
                    text="",
                    markdown=markdown,
                    tables=tables,
                    metadata={"processor": "docling", "file_name": file_name},
                    content_type="application/pdf",
                    warnings=warnings,
                )
            except Exception as exc:
                return self._fallback_bytes(data, file_name, warnings=[f"Docling failed: {exc}"])
        return self._fallback_bytes(data, file_name)

    # ------------------------------ internals ------------------------------
    def _fallback_path(self, path: Path, warnings: Optional[List[str]] = None) -> FileProcessingResult:
        with path.open("rb") as fh:
            stream = io.BytesIO(fh.read())
        text = _extract_text_pdfminer(stream)
        markdown = self._to_markdown(text)
        tables: List[Dict[str, Any]] = []
        if self.extract_tables:
            for df in _extract_tables_camelot(path):
                try:
                    tables.append({"shape": list(df.shape), "data": df.to_dict(orient="split")})
                except Exception:
                    continue
        return FileProcessingResult(
            success=True,
            text=text,
            markdown=markdown,
            tables=tables,
            metadata={"processor": "fallback", "file_name": path.name},
            content_type="application/pdf",
            warnings=warnings or [],
        )

    def _fallback_bytes(
        self, data: bytes, file_name: str, warnings: Optional[List[str]] = None
    ) -> FileProcessingResult:
        text = _extract_text_pdfminer(io.BytesIO(data))
        markdown = self._to_markdown(text)
        tables: List[Dict[str, Any]] = []
        if self.extract_tables:
            try:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
                    tmp.write(data)
                    tmp.flush()
                    for df in _extract_tables_camelot(Path(tmp.name)):
                        try:
                            tables.append({"shape": list(df.shape), "data": df.to_dict(orient="split")})
                        except Exception:
                            continue
            except Exception:
                pass
        return FileProcessingResult(
            success=True,
            text=text,
            markdown=markdown,
            tables=tables,
            metadata={"processor": "fallback", "file_name": file_name},
            content_type="application/pdf",
            warnings=warnings or [],
        )

    def _docling_convert_path(self, path: Path) -> Tuple[str, List[Dict[str, Any]], List[str]]:
        warnings: List[str] = []
        docling = self._docling
        try:
            if hasattr(docling, "convert"):
                result = docling.convert(str(path))  # type: ignore[attr-defined]
                markdown = result.get("markdown") if isinstance(result, dict) else str(result)
            else:
                markdown = _extract_text_pdfminer(io.BytesIO(path.read_bytes()))
                warnings.append("Docling module found but unsupported API; used fallback text extraction")
        except Exception as exc:
            markdown = _extract_text_pdfminer(io.BytesIO(path.read_bytes()))
            warnings.append(f"Docling conversion error: {exc}")
        tables: List[Dict[str, Any]] = []
        if self.extract_tables:
            for df in _extract_tables_camelot(path):
                try:
                    tables.append({"shape": list(df.shape), "data": df.to_dict(orient="split")})
                except Exception:
                    continue
        return markdown if isinstance(markdown, str) else str(markdown), tables, warnings

    def _docling_convert_bytes(self, data: bytes) -> Tuple[str, List[Dict[str, Any]], List[str]]:
        warnings: List[str] = []
        docling = self._docling
        try:
            if hasattr(docling, "convert_bytes"):
                result = docling.convert_bytes(data)  # type: ignore[attr-defined]
                markdown = result.get("markdown") if isinstance(result, dict) else str(result)
            else:
                markdown = _extract_text_pdfminer(io.BytesIO(data))
                warnings.append("Docling module found but unsupported API; used fallback text extraction")
        except Exception as exc:
            markdown = _extract_text_pdfminer(io.BytesIO(data))
            warnings.append(f"Docling conversion error: {exc}")
        tables: List[Dict[str, Any]] = []
        # For bytes table extraction, write to temp and reuse Camelot if available
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
                tmp.write(data)
                tmp.flush()
                for df in _extract_tables_camelot(Path(tmp.name)):
                    try:
                        tables.append({"shape": list(df.shape), "data": df.to_dict(orient="split")})
                    except Exception:
                        continue
        except Exception:
            pass
        return markdown if isinstance(markdown, str) else str(markdown), tables, warnings

    def _to_markdown(self, text: str) -> str:
        if not text:
            return ""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        return "\n\n".join(paragraphs)

