"""Docling-based PDF processor with robust fallbacks (Task 1.3.1).

Attempts to use Docling for structured markdown conversion. Falls back to
pdfminer.six/PyPDF2 for text and Camelot for table extraction when available.
"""
from __future__ import annotations

import io
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import zlib

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


_STREAM_RE = re.compile(rb"stream[\r\n]+(.*?)[\r\n]+endstream", re.DOTALL)


def _maybe_decompress(data: bytes) -> bytes:
    for wbits in (zlib.MAX_WBITS, -15):
        try:
            return zlib.decompress(data, wbits)
        except zlib.error:
            continue
    return data


def _decode_pdf_text(data: bytes) -> List[str]:
    texts: List[str] = []
    length = len(data)
    i = 0

    while i < length:
        byte = data[i]
        if byte == 0x28:  # '('
            depth = 1
            i += 1
            buf = bytearray()
            escape = False

            while i < length and depth > 0:
                b = data[i]
                i += 1

                if escape:
                    if b in (ord("n"), ord("r"), ord("t"), ord("b"), ord("f")):
                        mapping = {
                            ord("n"): "\n",
                            ord("r"): "\r",
                            ord("t"): "\t",
                            ord("b"): "\b",
                            ord("f"): "\f",
                        }
                        buf.extend(mapping[b].encode("utf-8"))
                    elif b in (ord("("), ord(")"), ord("\\")):
                        buf.append(b)
                    elif 48 <= b <= 55:  # octal escape
                        oct_digits = [b]
                        for _ in range(2):
                            if i < length and 48 <= data[i] <= 55:
                                oct_digits.append(data[i])
                                i += 1
                            else:
                                break
                        buf.append(int(bytes(oct_digits), 8))
                        i -= 1
                    else:
                        buf.append(b)
                    escape = False
                    continue

                if b == 0x5C:  # '\\'
                    escape = True
                    continue

                if b == 0x28:
                    depth += 1
                    buf.append(b)
                    continue

                if b == 0x29:
                    depth -= 1
                    if depth == 0:
                        try:
                            texts.append(buf.decode("utf-8", errors="ignore"))
                        except Exception:
                            texts.append(buf.decode("latin-1", errors="ignore"))
                        break
                    buf.append(b)
                    continue

                buf.append(b)
            continue

        i += 1

    return texts


def _extract_text_basic(data: bytes) -> str:
    snippets: List[str] = []
    for match in _STREAM_RE.finditer(data):
        raw = match.group(1)
        candidate = _maybe_decompress(raw)
        snippets.extend(_decode_pdf_text(candidate))
    combined = "\n".join(part for part in snippets if part.strip())
    return combined


def _extract_text_pdfminer(stream: io.BytesIO) -> Tuple[str, List[str]]:
    warnings: List[str] = []
    try:
        from pdfminer.high_level import extract_text  # type: ignore

        stream.seek(0)
        return extract_text(stream), warnings
    except Exception as exc:
        warnings.append(f"pdfminer text extraction unavailable: {exc}")
        try:
            import PyPDF2  # type: ignore

            stream.seek(0)
            reader = PyPDF2.PdfReader(stream)
            parts = [(page.extract_text() or "") for page in reader.pages]
            text = "\n".join(parts)
            if text.strip():
                return text, warnings
            warnings.append("PyPDF2 returned no text; attempting lightweight fallback")
        except Exception as exc2:
            warnings.append(f"PyPDF2 text extraction unavailable: {exc2}")

    stream.seek(0)
    basic = _extract_text_basic(stream.read())
    if basic.strip():
        warnings.append("Used lightweight PDF text fallback; install docling/pdfminer for richer extraction")
    return basic, warnings


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
        text, extraction_warnings = _extract_text_pdfminer(stream)
        all_warnings = list(warnings or [])
        all_warnings.extend(extraction_warnings)
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
            warnings=all_warnings,
        )

    def _fallback_bytes(
        self, data: bytes, file_name: str, warnings: Optional[List[str]] = None
    ) -> FileProcessingResult:
        text, extraction_warnings = _extract_text_pdfminer(io.BytesIO(data))
        all_warnings = list(warnings or [])
        all_warnings.extend(extraction_warnings)
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
            warnings=all_warnings,
        )

    def _docling_convert_path(self, path: Path) -> Tuple[str, List[Dict[str, Any]], List[str]]:
        warnings: List[str] = []
        docling = self._docling
        try:
            if hasattr(docling, "convert"):
                result = docling.convert(str(path))  # type: ignore[attr-defined]
                markdown = result.get("markdown") if isinstance(result, dict) else str(result)
            else:
                markdown_text, extra_warnings = _extract_text_pdfminer(io.BytesIO(path.read_bytes()))
                markdown = markdown_text
                warnings.append("Docling module found but unsupported API; used fallback text extraction")
                warnings.extend(extra_warnings)
        except Exception as exc:
            markdown_text, extra_warnings = _extract_text_pdfminer(io.BytesIO(path.read_bytes()))
            markdown = markdown_text
            warnings.append(f"Docling conversion error: {exc}")
            warnings.extend(extra_warnings)
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
                markdown_text, extra_warnings = _extract_text_pdfminer(io.BytesIO(data))
                markdown = markdown_text
                warnings.append("Docling module found but unsupported API; used fallback text extraction")
                warnings.extend(extra_warnings)
        except Exception as exc:
            markdown_text, extra_warnings = _extract_text_pdfminer(io.BytesIO(data))
            markdown = markdown_text
            warnings.append(f"Docling conversion error: {exc}")
            warnings.extend(extra_warnings)
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
