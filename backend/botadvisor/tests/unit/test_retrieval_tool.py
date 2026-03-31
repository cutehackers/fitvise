from __future__ import annotations

from unittest.mock import Mock

from botadvisor.app.core.entity.chunk import Chunk
from botadvisor.app.core.entity.document_metadata import DocumentMetadata


def make_tool_chunk(*, chunk_id: str, content: str, score: float) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        content=content,
        metadata=DocumentMetadata(
            doc_id="doc-1",
            source_id="file-1",
            platform="filesystem",
            source_url="file:///tmp/sample.txt",
            page=1,
            section="chunk_0",
        ),
        score=score,
    )


def test_retrieval_tool_maps_tool_input_to_retriever_request():
    from botadvisor.app.tools.retrieval_tool import build_retrieval_tool_definition

    retrieval_service = Mock()
    retrieval_service.retrieve.return_value = [make_tool_chunk(chunk_id="chunk-1", content="first chunk", score=0.91)]

    definition = build_retrieval_tool_definition(retrieval_service)
    result = definition.handler(definition.input_model(query="protein intake", platform="filesystem", top_k=3))

    retrieval_service.retrieve.assert_called_once()
    request = retrieval_service.retrieve.call_args.args[0]
    assert request.query == "protein intake"
    assert request.platform == "filesystem"
    assert request.top_k == 3

    assert result.tool_name == "retrieval"
    assert result.total_results == 1


def test_retrieval_tool_preserves_citation_metadata_in_tool_payload():
    from botadvisor.app.tools.retrieval_tool import build_retrieval_tool_definition

    retrieval_service = Mock()
    retrieval_service.retrieve.return_value = [make_tool_chunk(chunk_id="chunk-1", content="first chunk", score=0.91)]

    definition = build_retrieval_tool_definition(retrieval_service)
    result = definition.handler(definition.input_model(query="protein intake"))

    citation = result.results[0]
    assert citation.document_id == "doc-1"
    assert citation.chunk_id == "chunk-1"
    assert citation.metadata["source_url"] == "file:///tmp/sample.txt"
    assert citation.metadata["page"] == 1
    assert citation.metadata["section"] == "chunk_0"
