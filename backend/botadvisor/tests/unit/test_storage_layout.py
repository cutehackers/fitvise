from __future__ import annotations


def test_build_artifact_name_uses_checksum_document_id_and_extension():
    from botadvisor.app.storage.layout import build_artifact_name

    artifact_name = build_artifact_name(
        checksum="abcd1234efgh5678",
        document_id="doc-1",
        mime_type="application/pdf",
    )

    assert artifact_name == "abcd1234efgh5678_doc-1.pdf"


def test_build_artifact_name_omits_extension_when_unknown():
    from botadvisor.app.storage.layout import build_artifact_name

    artifact_name = build_artifact_name(
        checksum="abcd1234efgh5678",
        document_id="doc-2",
        mime_type="application/x-custom",
    )

    assert artifact_name == "abcd1234efgh5678_doc-2"


def test_build_checksum_prefix_uses_first_four_checksum_characters():
    from botadvisor.app.storage.layout import build_checksum_prefix

    assert build_checksum_prefix("abcd1234efgh5678") == "ab/cd"
