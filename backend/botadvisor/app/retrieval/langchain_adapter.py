"""LangChain adapter for the canonical BotAdvisor retrieval service."""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from botadvisor.app.core.entity.chunk import Chunk
from botadvisor.app.core.entity.retriever_request import RetrieverRequest


class LangChainRetrieverAdapter(BaseRetriever):
    """Thin LangChain wrapper around the canonical retrieval service."""

    retrieval_service: Any
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        chunks = self.retrieval_service.retrieve(RetrieverRequest(query=query))
        return [self._chunk_to_document(chunk) for chunk in chunks]

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.invoke(query))

    @staticmethod
    def _chunk_to_document(chunk: Chunk) -> Document:
        return Document(
            page_content=chunk.content,
            metadata={
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.metadata.doc_id,
                "source_id": chunk.metadata.source_id,
                "platform": chunk.metadata.platform,
                "source_url": chunk.metadata.source_url,
                "page": chunk.metadata.page,
                "section": chunk.metadata.section,
                "_distance": chunk.score,
            },
        )
