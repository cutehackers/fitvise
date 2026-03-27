"""Text normalization pipeline using ``SpacyTextProcessor`` (Task 1.3.2).

This module provides a small use case layer around the infrastructure
processor to normalize raw text content prior to downstream RAG steps.
It focuses on making text consistent and model-friendly while keeping
dependencies optional and failures non-fatal.

Key capabilities
- Unicode/case normalization and whitespace standardization
- Optional typo correction and lemmatization
- Optional named entity extraction (NER)
- Environment reporting so callers can branch on availability

All external dependencies (e.g., spaCy, textblob) are optional. When a
dependency is missing, the pipeline gracefully degrades to lightweight
string normalization so higher-level flows remain usable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from app.infrastructure.external_services.data_sources.file_processors import (
    SpacyTextProcessor,
    NormalizeTextOptions as ProcessorCleanTextOptions,
)


def _lib_available(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False


@dataclass
class NormalizedText:
    """Container for a single text's normalization result.

    Attributes
    - original: Input text provided by the caller.
    - cleaned: Normalized/standardized text suitable for downstream steps.
    - tokens: Tokenized representation (if available from the processor).
    - lemmas: Lemmatized tokens (if enabled/available).
    - entities: Extracted entities as dicts with type/offsets (if enabled/available).
    """
    original: str
    normalized: str
    tokens: List[str]
    lemmas: List[str]
    entities: List[Dict[str, Any]]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "original": self.original,
            "normalized": self.normalized,
            "tokens": self.tokens,
            "lemmas": self.lemmas,
            "entities": self.entities,
        }


class NormalizeTextRequest:
    """Request options for text normalization.

    Args
    - texts: One or more raw input texts.
    - lowercase: If True, apply case folding.
    - correct_typos: If True and supported, attempt light typo correction.
    - lemmatize: If True and supported, return lemmas alongside tokens.
    - extract_entities: If True and supported, include named entities.
    """
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


class NormalizeTextResponse:
    """Response object returned by ``NormalizeTextUseCase``.

    Attributes
    - results: A list of ``NormalizedText`` items in the same order as inputs.
    - environment: Capability flags (e.g., library/model availability) to help callers
      make decisions when features are unavailable.
    """
    def __init__(self, results: List[NormalizedText], environment: Dict[str, Any]) -> None:
        self.results = results
        self.environment = environment

    def as_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.as_dict() for r in self.results],
            "environment": self.environment,
        }


class NormalizeTextUseCase:
    """Normalize raw texts for consistent downstream processing.

    This use case coordinates normalization using ``SpacyTextProcessor`` and
    provides a stable interface for the API layer and pipelines.

    Behavior
    - Applies normalization to each input string.
    - Optionally performs typo correction, lemmatization, and NER depending on
      request flags and available libraries/models.
    - Always returns environment information so callers can branch on features.

    Args
    - spacy_model: Optional spaCy model name to load (e.g., "en_core_web_sm"). If
      not provided or not available, the processor falls back to minimal rules.

    Returns
    - ``NormalizeTextResponse`` with per-text results and environment flags.

    Example
    >>> uc = NormalizeTextUseCase("en_core_web_sm")
    >>> req = NormalizeTextRequest(texts=["Dr. Smith's papers..."], lowercase=True, lemmatize=True)
    >>> res = await uc.execute(req)
    >>> res.results[0].normalized  # normalized text
    "dr. smith's papers..."
    """
    def __init__(self, spacy_model: Optional[str] = None) -> None:
        self._spacy_model = spacy_model
        self._processor = SpacyTextProcessor(spacy_model)

    async def execute(self, request: NormalizeTextRequest) -> NormalizeTextResponse:
        results: List[NormalizedText] = []
        for text in request.texts:
            options = ProcessorCleanTextOptions(
                lowercase=request.lowercase,
                correct_typos=request.correct_typos,
                lemmatize=request.lemmatize,
                extract_entities=request.extract_entities,
            )
            processed = self._processor.normalize_text(text, options)
            results.append(
                NormalizedText(
                    original=text,
                    normalized=processed.normalized,
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
        return NormalizeTextResponse(results=results, environment=env)
