from __future__ import annotations

from typing import Protocol

from ..config import AppConfig
from ..domain.models import QueryIntent, SearchResult
from .dense import DenseRetriever
from .intent import build_retrieval_query, classify_query


class RetrievalBackend(Protocol):
    def search(self, query: str, limit: int) -> list[SearchResult]: ...


class Retriever:
    """Understand a query, then delegate evidence lookup to one backend."""

    def __init__(self, config: AppConfig, *, backend: RetrievalBackend | None = None):
        self.config = config
        self.backend = backend or DenseRetriever(config)
        self.last_intent = QueryIntent()
        self.last_search_query = ""

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        clean_query = " ".join(query.split())
        if not clean_query:
            self.last_intent = QueryIntent()
            self.last_search_query = ""
            return []

        self.last_intent = classify_query(clean_query)
        self.last_search_query = build_retrieval_query(clean_query, self.last_intent)
        limit = self.config.retrieval.top_k if top_k is None else top_k
        return self.backend.search(self.last_search_query, limit)
