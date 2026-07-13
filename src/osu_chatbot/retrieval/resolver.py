from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import math
from typing import Any, Iterable

from ..domain.artifacts import (
    CHUNKS_FILE,
    DOCUMENT_ALIASES_FILE,
    DOCUMENTS_FILE,
    LINK_ALIAS_CANDIDATES_FILE,
    load_chunks,
    load_records,
    read_jsonl,
)
from ..domain.models import Chunk, QueryIntent
from ..domain.source_profile import profile_for_document
from ..knowledge.aliases import build_document_alias_rows, normalize_alias, query_alias_keys
from ..knowledge.topics import load_or_build_canonical_topic_rows, topic_rows_by_document, topic_rows_by_equivalent_id
from .lexical import document_id, section_text_parts, tokenize


SOURCE_WEIGHTS = {
    "title": 22.0,
    "page_id": 18.0,
    "path_tail": 16.0,
    "accepted_link": 15.0,
    "topic_identifier": 17.0,
    "topic_alias": 14.0,
    "title_acronym": 24.0,
    "chunk_acronym": 24.0,
    "tag": 4.0,
    "heading": 3.0,
}


@dataclass(frozen=True)
class AliasEntry:
    alias: str
    target_document_id: str
    source: str
    confidence: float = 1.0
    retrieval_lane: str = "community"
    trust_tier: str = "unverified"
    topic_id: str = ""
    topic_document_ids: tuple[str, ...] = ()


@dataclass
class DocumentCandidate:
    document_id: str
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    matched_aliases: list[str] = field(default_factory=list)
    retrieval_lane: str = "community"
    trust_tier: str = "unverified"
    topic_id: str = ""
    topic_document_ids: list[str] = field(default_factory=list)

    def add(self, score: float, reason: str, alias: str | None = None) -> None:
        self.score += score
        self.reasons.append(reason)
        if alias:
            self.matched_aliases.append(alias)


class DocumentResolver:
    def __init__(
        self,
        documents: dict[str, dict[str, Any]],
        artifact_dir: Path,
        *,
        chunks: Iterable[Chunk] | None = None,
        aliases: list[dict[str, Any]] | None = None,
    ):
        self.documents = documents
        self.artifact_dir = artifact_dir
        self.chunks = list(chunks or [])
        self.topic_rows = load_or_build_canonical_topic_rows(artifact_dir, documents.values())
        self._topics_by_document = topic_rows_by_document(self.topic_rows)
        self._topics_by_equivalent_id = topic_rows_by_equivalent_id(self.topic_rows)
        self.alias_rows = aliases if aliases is not None else self._load_or_build_alias_rows()
        self._document_profiles = self._build_document_profiles()
        self._alias_index: dict[str, list[AliasEntry]] = defaultdict(list)
        self._document_token_counts: dict[str, Counter[str]] = {}
        self._title_tokens: dict[str, set[str]] = {}
        self._tail_tokens: dict[str, set[str]] = {}
        self._build_alias_index()

    def resolve(self, query: str, intent: QueryIntent, limit: int) -> list[DocumentCandidate]:
        if not self.documents:
            return []

        candidates: dict[str, DocumentCandidate] = {}
        full_key = normalize_alias(query)
        alias_matches: dict[tuple[str, str], tuple[float, AliasEntry]] = {}
        for key in query_alias_keys(query):
            for entry in self._alias_index.get(key, []):
                base = SOURCE_WEIGHTS.get(entry.source, 1.0) * entry.confidence
                if key == full_key:
                    base *= 1.35
                elif " " in key:
                    base *= 1.12
                if entry.source in {"title_acronym", "chunk_acronym"} and len(key) <= 4:
                    base *= 1.5
                match_key = (entry.target_document_id, key)
                previous = alias_matches.get(match_key)
                if previous is None or base > previous[0]:
                    alias_matches[match_key] = (base, entry)

        for (_doc_id, _key), (score, entry) in alias_matches.items():
            candidate = candidates.setdefault(
                entry.target_document_id,
                DocumentCandidate(
                    entry.target_document_id,
                    retrieval_lane=entry.retrieval_lane,
                    trust_tier=entry.trust_tier,
                    topic_id=entry.topic_id,
                    topic_document_ids=list(entry.topic_document_ids),
                ),
            )
            candidate.add(score, entry.source, entry.alias)

        query_terms = Counter([*tokenize(query), *intent.expanded_terms])
        for doc_id, hint in intent.document_hints.items():
            if doc_id in self.documents:
                candidate = candidates.setdefault(doc_id, self._candidate_for(doc_id))
                candidate.add(hint * 3.0, "intent")

        for doc_id, document in self.documents.items():
            score = self._token_overlap_score(doc_id, query_terms)
            if score:
                candidate = candidates.setdefault(doc_id, self._candidate_for(doc_id))
                candidate.add(score, "token_overlap")
                if intent.has("troubleshooting") and str(document.get("subculture") or "") == "help":
                    candidate.add(2.0, "intent_help")
                if intent.has("definition"):
                    title_key = normalize_alias(str(document.get("title") or ""))
                    if title_key and title_key in full_key:
                        candidate.add(2.0, "definition_title")

        ranked = sorted(
            candidates.values(),
            key=lambda candidate: (-candidate.score, candidate.document_id.count("/"), candidate.document_id),
        )
        return ranked[:limit]

    def accepted_alias_target(self, alias: str) -> str | None:
        key = normalize_alias(alias)
        entries = [entry for entry in self._alias_index.get(key, []) if entry.source == "accepted_link"]
        if not entries:
            return None
        return max(entries, key=lambda entry: entry.confidence).target_document_id

    def _candidate_for(self, doc_id: str) -> DocumentCandidate:
        lane, trust = self._document_profiles.get(doc_id, ("community", "unverified"))
        topic_id, topic_document_ids = self.topic_for_document(doc_id)
        return DocumentCandidate(
            doc_id,
            retrieval_lane=lane,
            trust_tier=trust,
            topic_id=topic_id,
            topic_document_ids=topic_document_ids,
        )

    def topic_for_document(self, doc_id: str) -> tuple[str, list[str]]:
        topic = self._topics_by_document.get(doc_id) or self._topics_by_equivalent_id.get(doc_id)
        if not topic:
            return doc_id, [doc_id]
        topic_id = str(topic.get("topic_id") or doc_id)
        equivalent_ids = [str(value) for value in topic.get("equivalent_document_ids", []) if str(value)]
        if topic_id not in equivalent_ids:
            equivalent_ids.append(topic_id)
        if doc_id not in equivalent_ids:
            equivalent_ids.append(doc_id)
        return topic_id, sorted(set(equivalent_ids), key=lambda value: (value.casefold(), value))

    def _build_document_profiles(self) -> dict[str, tuple[str, str]]:
        profiles: dict[str, tuple[str, str]] = {}
        for row in self.alias_rows:
            doc_id = str(row.get("document_id") or row.get("canonical_document_id") or "")
            if doc_id and doc_id not in profiles:
                profiles[doc_id] = (
                    str(row.get("retrieval_lane") or "community"),
                    str(row.get("trust_tier") or "unverified"),
                )
        for doc_id, document in self.documents.items():
            if doc_id not in profiles:
                profile = profile_for_document(document)
                profiles[doc_id] = (profile.lane, profile.trust_tier)
        return profiles

    def _load_or_build_alias_rows(self) -> list[dict[str, Any]]:
        alias_path = self.artifact_dir / DOCUMENT_ALIASES_FILE
        rows = read_jsonl(alias_path)
        if rows:
            return rows
        documents = list(self.documents.values())
        link_alias_rows = read_jsonl(self.artifact_dir / LINK_ALIAS_CANDIDATES_FILE)
        return build_document_alias_rows(documents, self.chunks, link_alias_rows, topic_rows=self.topic_rows)

    def _build_alias_index(self) -> None:
        for row in self.alias_rows:
            target = str(row.get("document_id") or row.get("canonical_document_id") or "")
            if target not in self.documents:
                target = str(row.get("canonical_document_id") or "")
            key = str(row.get("alias_key") or "")
            if not key or target not in self.documents:
                continue
            topic_id, topic_document_ids = self.topic_for_document(target)
            topic_id = str(row.get("topic_id") or topic_id)
            raw_topic_document_ids = row.get("topic_document_ids") or topic_document_ids
            topic_document_ids = sorted(
                {str(value) for value in raw_topic_document_ids if str(value)},
                key=lambda value: (value.casefold(), value),
            )
            self._alias_index[key].append(
                AliasEntry(
                    alias=str(row.get("alias") or key),
                    target_document_id=target,
                    source=str(row.get("source") or "alias"),
                    confidence=float(row.get("confidence") or 1.0),
                    retrieval_lane=str(row.get("retrieval_lane") or "community"),
                    trust_tier=str(row.get("trust_tier") or "unverified"),
                    topic_id=topic_id,
                    topic_document_ids=tuple(topic_document_ids),
                )
            )

    def _token_overlap_score(self, doc_id: str, query_terms: Counter[str]) -> float:
        if not query_terms:
            return 0.0
        doc_terms = self._document_terms_for(doc_id)
        overlap = sum(min(count, doc_terms.get(term, 0)) for term, count in query_terms.items())
        if not overlap:
            return 0.0
        title_overlap = sum(1 for term in query_terms if term in self._title_tokens_for(doc_id))
        tail_overlap = sum(1 for term in query_terms if term in self._tail_tokens_for(doc_id))
        score = math.log1p(overlap) * 1.8 + title_overlap * 2.8 + tail_overlap * 2.2
        source = str(self.documents[doc_id].get("source") or "")
        if source == "osu-news":
            score *= 0.65
        return score

    def _document_terms_for(self, doc_id: str) -> Counter[str]:
        if doc_id not in self._document_token_counts:
            document = self.documents[doc_id]
            text = " ".join(
                [
                    str(document.get("title") or ""),
                    doc_id.replace("_", " ").replace("/", " "),
                    " ".join(str(tag) for tag in document.get("tags", []) if isinstance(document.get("tags"), list)),
                    " ".join(str(tag) for tag in document.get("series_tags", []) if isinstance(document.get("series_tags"), list)),
                    str(document.get("domain") or ""),
                    str(document.get("subculture") or ""),
                    str(document.get("topic") or ""),
                    str(document.get("slug") or "").replace("-", " "),
                    " ".join(section_text_parts(document)),
                    " ".join(document_body_parts(document)),
                ]
            )
            self._document_token_counts[doc_id] = Counter(tokenize(text))
        return self._document_token_counts[doc_id]

    def _title_tokens_for(self, doc_id: str) -> set[str]:
        if doc_id not in self._title_tokens:
            self._title_tokens[doc_id] = set(tokenize(str(self.documents[doc_id].get("title") or "")))
        return self._title_tokens[doc_id]

    def _tail_tokens_for(self, doc_id: str) -> set[str]:
        if doc_id not in self._tail_tokens:
            self._tail_tokens[doc_id] = set(tokenize(doc_id.rsplit("/", 1)[-1].replace("_", " ")))
        return self._tail_tokens[doc_id]


def build_document_resolver(artifact_dir: Path) -> DocumentResolver:
    documents = {document_id(doc): doc for doc in load_records(artifact_dir / DOCUMENTS_FILE) if document_id(doc)}
    chunks = load_chunks(artifact_dir / CHUNKS_FILE)
    return DocumentResolver(documents, artifact_dir, chunks=chunks)


def document_body_parts(document: dict[str, Any], *, max_sections: int = 8, max_chars: int = 600) -> list[str]:
    sections = document.get("sections")
    if not isinstance(sections, list):
        return []
    parts: list[str] = []
    for section in sections[:max_sections]:
        if not isinstance(section, dict):
            continue
        text = str(section.get("text") or "")
        if text:
            parts.append(text[:max_chars])
    return parts
