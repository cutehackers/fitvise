from __future__ import annotations


def test_detect_mime_type_returns_expected_mapping(tmp_path):
    from botadvisor.app.ingestion.file_metadata import detect_mime_type

    file_path = tmp_path / "sample.md"
    file_path.write_text("# hello", encoding="utf-8")

    assert detect_mime_type(file_path) == "text/markdown"
