from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..config import AppConfig
from ..domain.models import SearchResult
from .analysis import ArtifactTopicResolver, DefaultQueryAnalyzer, QueryAnalyzer
from .dense import DenseRetriever
from .intent import build_retrieval_query
from .models import QueryAnalysis, RetrievalRequest


class RetrievalBackend(Protocol):
    def search(self, request: RetrievalRequest) -> list[SearchResult]: ...


@dataclass(frozen=True)
class RetrievalOutcome:
    results: list[SearchResult]
    analysis: QueryAnalysis
    search_query: str

    @property
    def intent(self):
        return self.analysis.intent


class Retriever:
    """Understand a query, then delegate evidence lookup to one backend."""

    def __init__(
        self,
        config: AppConfig,
        *,
        backend: RetrievalBackend | None = None,
        analyzer: QueryAnalyzer | None = None,
    ):
        self.config = config
        self.backend = backend or DenseRetriever(config)
        if analyzer is None:
            artifact_dir = config.artifacts.source_path or config.artifacts.path
            resolver = ArtifactTopicResolver(
                artifact_dir / config.retrieval.alias_artifact,
                minimum_confidence=config.retrieval.alias_minimum_confidence,
                minimum_tokens=config.retrieval.alias_minimum_tokens,
            )
            analyzer = DefaultQueryAnalyzer(resolver)
        self.analyzer = analyzer

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        return self.retrieve(query, top_k=top_k).results

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalOutcome:
        clean_query = " ".join(query.split())
        if not clean_query:
            return RetrievalOutcome(
                results=[],
                analysis=QueryAnalysis(query=""),
                search_query="",
            )

        analysis = self.analyzer.analyze(clean_query)
        if analysis.requires_clarification:
            return RetrievalOutcome(
                results=[],
                analysis=analysis,
                search_query=clean_query,
            )
        search_query = build_retrieval_query(clean_query, analysis.intent)
        limit = self.config.retrieval.top_k if top_k is None else top_k
        source_types = (
            self.config.retrieval.temporal_source_types
            if analysis.retrieval_lane == "temporal"
            else self.config.retrieval.canonical_source_types
        )
        common_request = {
            "query": search_query,
            "excluded_chunk_types": self.config.retrieval.excluded_chunk_types,
        }
        results: list[SearchResult] = []
        strong_document_ids = analysis.strong_preferred_document_ids
        if strong_document_ids:
            results = self.backend.search(
                RetrievalRequest(
                    **common_request,
                    limit=min(limit, self.config.retrieval.preferred_document_limit),
                    document_ids=strong_document_ids,
                )
            )
        soft_document_ids = analysis.soft_preferred_document_ids
        if soft_document_ids and len(results) < limit:
            soft_results = self.backend.search(
                RetrievalRequest(
                    **common_request,
                    limit=min(
                        limit - len(results),
                        self.config.retrieval.soft_preferred_document_limit,
                    ),
                    document_ids=soft_document_ids,
                )
            )
            results = merge_results(results, soft_results, limit=limit)
        if len(results) < limit:
            general_results = self.backend.search(
                RetrievalRequest(
                    **common_request,
                    limit=limit,
                    source_types=source_types,
                )
            )
            results = merge_results(results, general_results, limit=limit)
        return RetrievalOutcome(
            results=results,
            analysis=analysis,
            search_query=search_query,
        )

    def is_ready(self) -> bool:
        readiness_check = getattr(self.backend, "is_ready", None)
        return True if readiness_check is None else bool(readiness_check())


def merge_results(
    preferred: list[SearchResult],
    fallback: list[SearchResult],
    *,
    limit: int,
) -> list[SearchResult]:
    merged: list[SearchResult] = []
    seen_chunk_ids: set[str] = set()
    for result in [*preferred, *fallback]:
        if result.chunk.id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(result.chunk.id)
        merged.append(result)
        if len(merged) >= limit:
            break
    return merged
