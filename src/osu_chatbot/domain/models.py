from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeAlias

Document: TypeAlias = dict[str, Any]


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    source_type: str
    file_path: str
    osu_url: str
    title: str
    text: str
    chunk_index: int
    heading_path: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    date: str | None = None
    series: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Entity:
    canonical: str
    aliases: list[str]
    sources: list[str]
    score: float = 1.0


@dataclass(frozen=True)
class QueryIntent:
    labels: set[str] = field(default_factory=set)
    expanded_terms: set[str] = field(default_factory=set)
    document_hints: dict[str, float] = field(default_factory=dict)

    def has(self, label: str) -> bool:
        return label in self.labels


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    dense_score: float
    keyword_score: float
    entity_score: float
    document_score: float = 0.0

    @property
    def score(self) -> float:
        return self.dense_score + self.keyword_score + self.entity_score + self.document_score


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    kind: str
    message: str
    artifact: str
    identifier: str

    def to_record(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "kind": self.kind,
            "message": self.message,
            "artifact": self.artifact,
            "identifier": self.identifier,
        }
