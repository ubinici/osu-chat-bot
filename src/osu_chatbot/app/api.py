from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from ..config import AppConfig, load_config
from .service import ChatService

logger = logging.getLogger(__name__)


class ConversationTurnRequest(BaseModel):
    user: str = Field(min_length=1, max_length=2000)
    assistant: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[ConversationTurnRequest] = Field(default_factory=list, max_length=8)


class SourceResponse(BaseModel):
    citation: int
    chunk_id: str
    document_id: str
    title: str
    url: str
    score: float


class TopicResponse(BaseModel):
    canonical_id: str
    matched_alias: str
    document_ids: list[str]
    confidence: float


class ClarificationResponse(BaseModel):
    reason: str
    prompt: str
    options: list[str]


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    intent: list[str]
    search_query: str
    latency_ms: int
    retrieval_lane: str
    resolved_topics: list[TopicResponse]
    response_type: str
    clarification: ClarificationResponse | None


def create_app(
    config: AppConfig | None = None,
    *,
    service: ChatService | None = None,
) -> FastAPI:
    chat_service = service or ChatService(config or load_config())
    app = FastAPI(title="osu! Chatbot API", version="0.1.0")

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readiness() -> dict[str, str]:
        try:
            ready = chat_service.is_ready()
        except Exception:
            logger.exception("Readiness check failed")
            ready = False
        if not ready:
            raise HTTPException(status_code=503, detail="Retrieval index is not ready.")
        return {"status": "ready"}

    @app.post("/v1/chat", response_model=ChatResponse)
    async def chat(payload: ChatRequest) -> ChatResponse:
        try:
            history = [(turn.user, turn.assistant) for turn in payload.history]
            if history:
                result = await run_in_threadpool(chat_service.ask, payload.question, history)
            else:
                result = await run_in_threadpool(chat_service.ask, payload.question)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Chat request failed")
            raise HTTPException(status_code=503, detail="Chat dependencies are unavailable.") from exc
        return ChatResponse(
            answer=result.answer,
            sources=[SourceResponse(**vars(source)) for source in result.sources],
            intent=result.intent,
            search_query=result.search_query,
            latency_ms=result.latency_ms,
            retrieval_lane=result.retrieval_lane,
            resolved_topics=[TopicResponse(**vars(topic)) for topic in result.resolved_topics or []],
            response_type=result.response_type,
            clarification=(
                ClarificationResponse(**vars(result.clarification))
                if result.clarification
                else None
            ),
        )

    return app
