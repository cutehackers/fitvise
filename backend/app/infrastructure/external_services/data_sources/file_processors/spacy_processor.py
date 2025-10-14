"""spaCy-based text processor (Task 1.3.2/1.3.3).

Provides normalization, lemmatization, and NER via spaCy (when present) with
typo correction via TextBlob (optional). Includes TF-IDF-based keyword
extraction with a fallback word-frequency approach.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence

from .base_processor import (
    NormalizeTextOptions,
    NormalizeTextResult,
    MetadataExtractionResult,
    MetadataExtractorBase,
    TextCleanerBase,
)


def _try_import_spacy():
    try:
        import spacy  # type: ignore

        return spacy
    except Exception:
        return None


def _try_import_textblob():
    try:
        from textblob import TextBlob  # type: ignore

        return TextBlob
    except Exception:
        return None


def _try_import_langdetect():
    try:
        from langdetect import detect  # type: ignore

        return detect
    except Exception:
        return None


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[\u201C\u201D]", '"', text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _basic_tokens(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text)


def _extract_dates(text: str) -> List[str]:
    patterns = [
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{2}/\d{2}/\d{4}\b",
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s*\d{4}\b",
    ]
    results: List[str] = []
    for p in patterns:
        results.extend(re.findall(p, text))
    return list(dict.fromkeys(results))


def _extract_authors(text: str) -> List[str]:
    matches = re.findall(r"\bBy\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text)
    return list(dict.fromkeys(matches))


def _top_keywords_tfidf(texts: List[str], top_k: int = 10) -> List[List[str]]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

        vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        results: List[List[str]] = []
        for row in range(matrix.shape[0]):
            row_data = matrix.getrow(row)
            indices = row_data.indices
            scores = row_data.data
            pairs = sorted(zip(indices, scores), key=lambda x: x[1], reverse=True)
            top_terms = [feature_names[i] for i, _ in pairs[:top_k]]
            results.append(top_terms)
        return results
    except Exception:
        results: List[List[str]] = []
        for text in texts:
            words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
            counts: Dict[str, int] = {}
            for w in words:
                counts[w] = counts.get(w, 0) + 1
            top_terms = [w for w, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_k]]
            results.append(top_terms)
        return results


class SpacyTextProcessor(TextCleanerBase, MetadataExtractorBase):
    """spaCy-backed text cleaning and metadata extraction."""

    def __init__(self, spacy_model: Optional[str] = None) -> None:
        self._spacy = _try_import_spacy()
        self._nlp = None
        if self._spacy is not None and spacy_model:
            try:
                self._nlp = self._spacy.load(spacy_model)
            except Exception:
                self._nlp = None
        self._textblob = _try_import_textblob()
        self._langdetect = _try_import_langdetect()

    def normalize_text(self, text: str, options: Optional[NormalizeTextOptions] = None) -> NormalizeTextResult:
        options = options or NormalizeTextOptions()
        normalized = _normalize_text(text)
        if options.lowercase:
            normalized = normalized.lower()
        if options.correct_typos and self._textblob is not None:
            try:
                normalized = str(self._textblob(normalized).correct())
            except Exception:
                pass

        if self._spacy is not None and self._nlp is None:
            try:
                self._nlp = self._spacy.blank("en")
            except Exception:
                self._nlp = None

        if self._nlp is not None:
            try:
                doc = self._nlp(normalized)
                tokens = [t.text for t in doc]
                lemmas = [t.lemma_ if options.lemmatize else t.text for t in doc]
                entities: List[Dict[str, Any]] = []
                if options.extract_entities and hasattr(doc, "ents"):
                    for ent in doc.ents:
                        entities.append({"text": ent.text, "label": ent.label_})
                return NormalizeTextResult(
                    success=True,
                    original=text,
                    normalized=normalized,
                    tokens=tokens,
                    lemmas=lemmas,
                    entities=entities,
                )
            except Exception:
                pass

        tokens = _basic_tokens(normalized)
        lemmas = tokens if options.lemmatize else tokens
        return NormalizeTextResult(
            success=True, original=text, normalized=normalized, tokens=tokens, lemmas=lemmas, entities=[]
        )

    def extract(self, texts: Sequence[str], top_k_keywords: int = 10) -> List[MetadataExtractionResult]:
        kw_lists = _top_keywords_tfidf(list(texts), top_k_keywords)
        results: List[MetadataExtractionResult] = []
        for idx, text in enumerate(texts):
            entities: List[Dict[str, str]] = []
            if self._spacy is not None:
                try:
                    if self._nlp is None:
                        self._nlp = self._spacy.blank("en")
                    doc = self._nlp(text)
                    for ent in getattr(doc, "ents", []):
                        entities.append({"text": ent.text, "label": ent.label_})
                except Exception:
                    entities = []
            dates = _extract_dates(text)
            authors = _extract_authors(text)
            language: Optional[str] = None
            if self._langdetect is not None:
                try:
                    language = self._langdetect(text)
                except Exception:
                    language = None
            results.append(
                MetadataExtractionResult(
                    success=True,
                    keywords=kw_lists[idx] if idx < len(kw_lists) else [],
                    entities=entities,
                    dates=dates,
                    authors=authors,
                    language=language,
                )
            )
        return results

