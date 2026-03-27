"""Thin FastAPI routes for the canonical BotAdvisor runtime."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from botadvisor.app.api.deps import get_chat_service, get_health_service
from botadvisor.app.chat.schemas import ChatRequest, ChatResponse, HealthResponse, QueryRequest, QueryResponse


router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health(health_service=Depends(get_health_service)) -> HealthResponse:
    """Return runtime health status."""
    return await health_service.get_status()


@router.post("/query", response_model=QueryResponse, tags=["query"])
async def query(payload: QueryRequest, chat_service=Depends(get_chat_service)) -> QueryResponse:
    """Return retrieved sources for a query."""
    return chat_service.query(payload)


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(payload: ChatRequest, chat_service=Depends(get_chat_service)) -> ChatResponse:
    """Return a retrieval-backed answer with citations."""
    return chat_service.chat(payload)
