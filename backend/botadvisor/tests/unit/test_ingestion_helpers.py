from __future__ import annotations

from botadvisor.app.core.entity.document import Document


def test_create_chunks_preserves_document_metadata():
    from botadvisor.app.ingestion.chunking import create_chunks

    document = Document.create(
        source_id="file-1",
        platform="filesystem",
        source_url="file:///tmp/sample.txt",
        content=b"abcdefghij",
        mime_type="text/plain",
        id="doc-1",
    )

    chunks = create_chunks("abcdefghij", document=document, chunk_size=4)

    assert [chunk.content for chunk in chunks] == ["abcd", "efgh", "ij"]
    assert [chunk.chunk_id for chunk in chunks] == ["doc-1_chunk_0", "doc-1_chunk_1", "doc-1_chunk_2"]
    assert all(chunk.metadata.doc_id == "doc-1" for chunk in chunks)
    assert all(chunk.metadata.platform == "filesystem" for chunk in chunks)


def test_detect_mime_type_returns_expected_mapping(tmp_path):
    from botadvisor.app.ingestion.files import detect_mime_type

    file_path = tmp_path / "sample.md"
    file_path.write_text("# hello", encoding="utf-8")

    assert detect_mime_type(file_path) == "text/markdown"
