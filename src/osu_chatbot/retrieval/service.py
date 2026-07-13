from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..config import AppConfig
from ..domain.models import QueryIntent, SearchResult
from .dense import DenseRetriever
from .intent import build_retrieval_query, classify_query


class RetrievalBackend(Protocol):
    def search(self, query: str, limit: int) -> list[SearchResult]: ...


@dataclass(frozen=True)
class RetrievalOutcome:
    results: list[SearchResult]
    intent: QueryIntent
    search_query: str


class Retriever:
    """Understand a query, then delegate evidence lookup to one backend."""

    def __init__(self, config: AppConfig, *, backend: RetrievalBackend | None = None):
        self.config = config
        self.backend = backend or DenseRetriever(config)

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        return self.retrieve(query, top_k=top_k).results

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalOutcome:
        clean_query = " ".join(query.split())
        if not clean_query:
            return RetrievalOutcome(results=[], intent=QueryIntent(), search_query="")

        intent = classify_query(clean_query)
        search_query = build_retrieval_query(clean_query, intent)
        limit = self.config.retrieval.top_k if top_k is None else top_k
        return RetrievalOutcome(
            results=self.backend.search(search_query, limit),
            intent=intent,
            search_query=search_query,
        )

    def is_ready(self) -> bool:
        readiness_check = getattr(self.backend, "is_ready", None)
        return True if readiness_check is None else bool(readiness_check())
