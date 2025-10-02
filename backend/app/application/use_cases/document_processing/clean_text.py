"""Text cleaning pipeline using SpacyTextProcessor (Task 1.3.2).

Leverages infrastructure processor for normalization, optional typo
correction, lemmatization, and NER. Provides safe fallbacks when
optional dependencies aren't installed.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from app.infrastructure.external_services.data_sources.file_processors import (
    SpacyTextProcessor,
    CleanTextOptions as ProcessorCleanTextOptions,
)


def _lib_available(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False


@dataclass
class CleanedText:
    original: str
    cleaned: str
    tokens: List[str]
    lemmas: List[str]
    entities: List[Dict[str, Any]]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "original": self.original,
            "cleaned": self.cleaned,
            "tokens": self.tokens,
            "lemmas": self.lemmas,
            "entities": self.entities,
        }


class CleanTextRequest:
    def __init__(
        self,
        texts: Sequence[str],
        lowercase: bool = False,
        correct_typos: bool = False,
        lemmatize: bool = True,
        extract_entities: bool = True,
    ) -> None:
        self.texts = list(texts)
        self.lowercase = lowercase
        self.correct_typos = correct_typos
        self.lemmatize = lemmatize
        self.extract_entities = extract_entities


class CleanTextResponse:
    def __init__(self, results: List[CleanedText], environment: Dict[str, Any]) -> None:
        self.results = results
        self.environment = environment

    def as_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.as_dict() for r in self.results],
            "environment": self.environment,
        }


class CleanTextUseCase:
    def __init__(self, spacy_model: Optional[str] = None) -> None:
        self._spacy_model = spacy_model
        self._processor = SpacyTextProcessor(spacy_model)

    async def execute(self, request: CleanTextRequest) -> CleanTextResponse:
        results: List[CleanedText] = []
        for text in request.texts:
            options = ProcessorCleanTextOptions(
                lowercase=request.lowercase,
                correct_typos=request.correct_typos,
                lemmatize=request.lemmatize,
                extract_entities=request.extract_entities,
            )
            processed = self._processor.clean_text(text, options)
            results.append(
                CleanedText(
                    original=text,
                    cleaned=processed.cleaned,
                    tokens=processed.tokens,
                    lemmas=processed.lemmas,
                    entities=processed.entities,
                )
            )

        env = {
            "spacy_available": _lib_available("spacy"),
            "spacy_model_loaded": bool(self._spacy_model),
            "textblob_available": _lib_available("textblob"),
        }
        return CleanTextResponse(results=results, environment=env)

