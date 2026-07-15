from __future__ import annotations

from dataclasses import dataclass, field

from ..domain.models import QueryIntent


@dataclass(frozen=True)
class ResolvedTopic:
    """A source-derived topic linked to one or more retrievable documents."""

    canonical_id: str
    matched_alias: str
    document_ids: tuple[str, ...]
    confidence: float
    source_type: str | None = None
    retrieval_lane: str | None = None
    preference_strength: str = "strong"


@dataclass(frozen=True)
class ClarificationRequest:
    """A conservative request for missing user context."""

    reason: str
    prompt: str
    options: tuple[str, ...] = ()


@dataclass(frozen=True)
class QueryAnalysis:
    """Provider-neutral understanding shared by retrieval and future routers."""

    query: str
    intent: QueryIntent = field(default_factory=QueryIntent)
    topics: tuple[ResolvedTopic, ...] = ()
    retrieval_lane: str = "canonical"
    clarification: ClarificationRequest | None = None

    @property
    def requires_clarification(self) -> bool:
        return self.clarification is not None

    @property
    def preferred_document_ids(self) -> tuple[str, ...]:
        return self.strong_preferred_document_ids + self.soft_preferred_document_ids

    @property
    def strong_preferred_document_ids(self) -> tuple[str, ...]:
        return self._preferred_document_ids("strong")

    @property
    def soft_preferred_document_ids(self) -> tuple[str, ...]:
        return self._preferred_document_ids("soft")

    def _preferred_document_ids(self, strength: str) -> tuple[str, ...]:
        document_ids: list[str] = []
        for topic in self.topics:
            if topic.preference_strength != strength:
                continue
            for document_id in topic.document_ids:
                if document_id and document_id not in document_ids:
                    document_ids.append(document_id)
        return tuple(document_ids)


@dataclass(frozen=True)
class RetrievalRequest:
    """Backend-neutral dense retrieval request with optional metadata policy."""

    query: str
    limit: int
    source_types: tuple[str, ...] = ()
    document_ids: tuple[str, ...] = ()
    excluded_chunk_types: tuple[str, ...] = ()
