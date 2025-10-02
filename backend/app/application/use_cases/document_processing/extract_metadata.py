"""Metadata extraction service using SpacyTextProcessor (Task 1.3.3).

Extracts keywords (TF-IDF), entities (spaCy if available), dates, authors,
and language detection when supported. Uses infrastructure processor with
lightweight fallbacks for missing dependencies.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from app.infrastructure.external_services.data_sources.file_processors import SpacyTextProcessor


def _lib_available(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False


@dataclass
class ExtractedMetadata:
    keywords: List[str]
    entities: List[Dict[str, str]]
    dates: List[str]
    authors: List[str]
    language: Optional[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "keywords": self.keywords,
            "entities": self.entities,
            "dates": self.dates,
            "authors": self.authors,
            "language": self.language,
        }


class ExtractMetadataRequest:
    def __init__(self, texts: Sequence[str], top_k_keywords: int = 10) -> None:
        self.texts = list(texts)
        self.top_k_keywords = top_k_keywords


class ExtractMetadataResponse:
    def __init__(self, results: List[ExtractedMetadata], environment: Dict[str, Any]) -> None:
        self.results = results
        self.environment = environment

    def as_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.as_dict() for r in self.results],
            "environment": self.environment,
        }


class ExtractMetadataUseCase:
    def __init__(self, spacy_model: Optional[str] = None) -> None:
        self._spacy_model = spacy_model
        self._processor = SpacyTextProcessor(spacy_model)

    async def execute(self, request: ExtractMetadataRequest) -> ExtractMetadataResponse:
        proc_results = self._processor.extract(request.texts, request.top_k_keywords)
        results: List[ExtractedMetadata] = [
            ExtractedMetadata(
                keywords=pr.keywords,
                entities=pr.entities,
                dates=pr.dates,
                authors=pr.authors,
                language=pr.language,
            )
            for pr in proc_results
        ]

        env = {
            "spacy_available": _lib_available("spacy"),
            "spacy_model_loaded": bool(self._spacy_model),
            "langdetect_available": _lib_available("langdetect"),
            "tfidf_vectorizer": True,
        }
        return ExtractMetadataResponse(results=results, environment=env)
