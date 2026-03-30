"""Deterministic fixture corpus for canonical retrieval evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from botadvisor.app.core.entity.chunk import Chunk
from botadvisor.app.core.entity.document import Document
from botadvisor.app.ingestion.text_chunking import create_chunks


EVALUATION_SOURCE_PREFIX = "evaluation_corpus"


@dataclass(frozen=True)
class EvaluationCorpusDocument:
    """A checked-in corpus document used for retrieval evaluation."""

    document_id: str
    source_id: str
    source_url: str
    text: str


def load_evaluation_corpus_documents(corpus_root: Path) -> list[EvaluationCorpusDocument]:
    """Load the checked-in retrieval evaluation corpus with deterministic identifiers."""
    documents: list[EvaluationCorpusDocument] = []

    for path in sorted(corpus_root.glob("*.md")):
        source_id = f"{EVALUATION_SOURCE_PREFIX}/{path.name}"
        documents.append(
            EvaluationCorpusDocument(
                document_id=path.stem,
                source_id=source_id,
                source_url=f"evaluation://corpus/{path.name}",
                text=path.read_text(encoding="utf-8"),
            )
        )

    return documents


def build_evaluation_chunks(corpus_root: Path) -> list[Chunk]:
    """Build chunk entities from the checked-in retrieval evaluation corpus."""
    chunks: list[Chunk] = []

    for corpus_document in load_evaluation_corpus_documents(corpus_root):
        document = Document.create(
            id=corpus_document.document_id,
            source_id=corpus_document.source_id,
            platform="filesystem",
            source_url=corpus_document.source_url,
            content=corpus_document.text.encode("utf-8"),
            mime_type="text/markdown",
        )
        chunks.extend(create_chunks(corpus_document.text, document=document))

    return chunks


def write_evaluation_chunks(chunks: list[Chunk], output_path: Path) -> Path:
    """Write evaluation chunks to a single JSON file for canonical upsert flow."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([chunk.to_dict() for chunk in chunks], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path
