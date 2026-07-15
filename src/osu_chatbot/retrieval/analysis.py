from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..domain.artifacts import read_jsonl
from .intent import TOKEN_RE, classify_query, normalize_token
from .models import ClarificationRequest, QueryAnalysis, ResolvedTopic

VAGUE_QUERIES = {
    "help",
    "help me",
    "it doesnt work",
    "it does not work",
    "this doesnt work",
    "this does not work",
    "how do i fix it",
    "what should i do",
    "where do i go",
    "can you explain this",
    "what is this",
    "what about that",
}
CLARIFICATION_OPTIONS = (
    "gameplay or difficulty",
    "client, audio, or performance",
    "beatmapping",
    "account, rules, or support",
)


class TopicResolver(Protocol):
    def resolve(self, query: str) -> tuple[ResolvedTopic, ...]: ...


class QueryAnalyzer(Protocol):
    def analyze(self, query: str) -> QueryAnalysis: ...


class NullTopicResolver:
    def resolve(self, query: str) -> tuple[ResolvedTopic, ...]:
        return ()


@dataclass(frozen=True)
class _AliasCandidate:
    canonical_id: str
    alias: str
    document_ids: tuple[str, ...]
    confidence: float
    source_type: str | None
    retrieval_lane: str | None
    preference_strength: str


class ArtifactTopicResolver:
    """Resolve query phrases through source-generated alias records.

    The JSONL schema is intentionally source-neutral. Scrapers can emit the same
    fields even when a document has no relationship to an osu! wiki page.
    """

    def __init__(
        self,
        path: Path,
        *,
        minimum_confidence: float = 0.85,
        minimum_tokens: int = 2,
    ):
        self.path = path
        self.minimum_confidence = minimum_confidence
        self.minimum_tokens = max(1, minimum_tokens)
        self._aliases = self._load_aliases(path)
        self._max_alias_tokens = min(6, max((len(key.split()) for key in self._aliases), default=1))

    def resolve(self, query: str) -> tuple[ResolvedTopic, ...]:
        raw_tokens = TOKEN_RE.findall(query)
        tokens = [normalize_token(token) for token in raw_tokens]
        resolved: list[ResolvedTopic] = []
        occupied: set[int] = set()

        for width in range(min(self._max_alias_tokens, len(tokens)), 0, -1):
            if width < self.minimum_tokens:
                continue
            for start in range(0, len(tokens) - width + 1):
                positions = set(range(start, start + width))
                if positions.intersection(occupied):
                    continue
                alias_key = " ".join(tokens[start : start + width])
                candidates = self._aliases.get(alias_key, ())
                candidate = self._select_candidate(candidates)
                if candidate is None:
                    continue
                resolved.append(
                    ResolvedTopic(
                        canonical_id=candidate.canonical_id,
                        matched_alias=" ".join(raw_tokens[start : start + width]),
                        document_ids=candidate.document_ids,
                        confidence=candidate.confidence,
                        source_type=candidate.source_type,
                        retrieval_lane=candidate.retrieval_lane,
                        preference_strength=candidate.preference_strength,
                    )
                )
                occupied.update(positions)

        return tuple(resolved)

    def _load_aliases(self, path: Path) -> dict[str, tuple[_AliasCandidate, ...]]:
        grouped: dict[str, list[_AliasCandidate]] = defaultdict(list)
        for record in read_jsonl(path):
            confidence = float(record.get("confidence") or 0.0)
            if confidence < self.minimum_confidence:
                continue
            alias = str(record.get("alias") or "").strip()
            alias_key = normalize_phrase(str(record.get("alias_key") or alias))
            document_id = str(record.get("document_id") or "").strip()
            canonical_id = str(
                record.get("canonical_document_id")
                or record.get("topic_id")
                or document_id
            ).strip()
            if not alias_key or not document_id or not canonical_id:
                continue
            topic_document_ids = record.get("topic_document_ids")
            document_ids = _unique_strings(
                [canonical_id, document_id]
                + (list(topic_document_ids) if isinstance(topic_document_ids, list) else [])
            )
            grouped[alias_key].append(
                _AliasCandidate(
                    canonical_id=canonical_id,
                    alias=alias,
                    document_ids=document_ids,
                    confidence=confidence,
                    source_type=_optional_string(record.get("target_source")),
                    retrieval_lane=_optional_string(record.get("retrieval_lane")),
                    preference_strength=_preference_strength(
                        record,
                        alias_key=alias_key,
                        canonical_id=canonical_id,
                    ),
                )
            )
        return {key: tuple(value) for key, value in grouped.items()}

    @staticmethod
    def _select_candidate(candidates: tuple[_AliasCandidate, ...]) -> _AliasCandidate | None:
        if not candidates:
            return None
        by_canonical: dict[str, list[_AliasCandidate]] = defaultdict(list)
        for candidate in candidates:
            by_canonical[candidate.canonical_id].append(candidate)
        ranked = sorted(
            (
                max(group, key=lambda item: item.confidence)
                for group in by_canonical.values()
            ),
            key=lambda item: item.confidence,
            reverse=True,
        )
        if len(ranked) > 1 and ranked[0].confidence - ranked[1].confidence < 0.15:
            return None
        winner = ranked[0]
        related_documents: list[str] = []
        for candidate in by_canonical[winner.canonical_id]:
            related_documents.extend(candidate.document_ids)
        preference_strength = (
            "strong"
            if any(
                candidate.preference_strength == "strong"
                for candidate in by_canonical[winner.canonical_id]
            )
            else "soft"
        )
        return _AliasCandidate(
            canonical_id=winner.canonical_id,
            alias=winner.alias,
            document_ids=_unique_strings(related_documents),
            confidence=winner.confidence,
            source_type=winner.source_type,
            retrieval_lane=winner.retrieval_lane,
            preference_strength=preference_strength,
        )


class DefaultQueryAnalyzer:
    def __init__(self, resolver: TopicResolver | None = None):
        self.resolver = resolver or NullTopicResolver()

    def analyze(self, query: str) -> QueryAnalysis:
        intent = classify_query(query)
        topics = self.resolver.resolve(query)
        return QueryAnalysis(
            query=query,
            intent=intent,
            topics=topics,
            retrieval_lane="temporal" if "temporal" in intent.labels else "canonical",
            clarification=_clarification_request(query, topics),
        )


def normalize_phrase(value: str) -> str:
    return " ".join(normalize_token(token) for token in TOKEN_RE.findall(value))


def _clarification_request(
    query: str,
    topics: tuple[ResolvedTopic, ...],
) -> ClarificationRequest | None:
    if topics or normalize_phrase(query) not in VAGUE_QUERIES:
        return None
    return ClarificationRequest(
        reason="missing_topic",
        prompt=(
            "What part of osu! do you need help with? For example: gameplay, "
            "the client, beatmapping, or your account."
        ),
        options=CLARIFICATION_OPTIONS,
    )


def _unique_strings(values: list[object]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _optional_string(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _preference_strength(
    record: dict[str, object],
    *,
    alias_key: str,
    canonical_id: str,
) -> str:
    explicit = str(record.get("preference_strength") or "").strip().casefold()
    if explicit in {"strong", "soft"}:
        return explicit
    source = str(record.get("source") or "").strip().casefold()
    if source in {"title", "path_tail", "page_id", "topic_identifier"}:
        return "strong"
    alias_tokens = set(normalize_phrase(alias_key).split())
    identifier = canonical_id.rstrip("/").rsplit("/", 1)[-1].replace("_", " ")
    identifier_tokens = set(normalize_phrase(identifier).split())
    if alias_tokens and alias_tokens.issubset(identifier_tokens):
        return "strong"
    return "soft"
