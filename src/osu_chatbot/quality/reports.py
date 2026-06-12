from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import re

from ..domain.artifacts import (
    CHUNKS_FILE,
    DOCUMENTS_FILE,
    STATS_REPORT_FILE,
    TERMS_FILE,
    VALIDATION_REPORT_FILE,
    load_chunks,
    load_entities,
    load_records,
    write_json,
)
from ..domain.models import ValidationIssue


def build_stats_report(artifact_dir: Path) -> dict[str, Any]:
    documents = load_records(artifact_dir / DOCUMENTS_FILE)
    chunks = load_chunks(artifact_dir / CHUNKS_FILE)
    entities = load_entities(artifact_dir / TERMS_FILE) if (artifact_dir / TERMS_FILE).exists() else []

    document_sources = Counter(str(doc.get("source") or "unknown") for doc in documents)
    document_domains = Counter(str(doc.get("domain") or "unknown") for doc in documents)
    document_subcultures = Counter(str(doc.get("subculture") or "unknown") for doc in documents)
    news_years = Counter(str(doc.get("year") or "unknown") for doc in documents if doc.get("source") == "osu-news")
    news_series = Counter(
        str(doc.get("series_primary") or "none") for doc in documents if doc.get("source") == "osu-news"
    )
    news_style_issues = Counter(
        str(issue)
        for doc in documents
        if doc.get("source") == "osu-news"
        for issue in _as_list(doc.get("style_issues"))
    )
    chunk_types = Counter(str(chunk.metadata.get("chunk_type") or "unknown") for chunk in chunks)
    chunk_sources = Counter(str(chunk.metadata.get("source") or chunk.source_type) for chunk in chunks)
    empty_sections = sum(
        1
        for doc in documents
        for section in _as_list(doc.get("sections"))
        if isinstance(section, dict) and not str(section.get("text") or "").strip()
    )

    chunk_lengths = sorted(
        (
            {
                "chunk_id": chunk.id,
                "title": chunk.title,
                "chunk_type": chunk.metadata.get("chunk_type"),
                "source": chunk.metadata.get("source") or chunk.source_type,
                "chars": len(chunk.text),
                "words": len(chunk.text.split()),
                "url": chunk.osu_url,
            }
            for chunk in chunks
        ),
        key=lambda item: item["chars"],
        reverse=True,
    )

    report = {
        "documents": {
            "total": len(documents),
            "by_source": _counter(document_sources),
            "by_domain_top20": _counter(document_domains, limit=20),
            "by_subculture": _counter(document_subcultures),
            "news_by_year": _counter(news_years),
            "news_by_series_top20": _counter(news_series, limit=20),
            "news_style_issues": _counter(news_style_issues),
        },
        "structure": {
            "sections": sum(_as_int(doc.get("section_count")) for doc in documents),
            "empty_sections": empty_sections,
            "links": sum(_as_int(doc.get("edge_count")) for doc in documents),
            "citations": sum(_as_int(doc.get("citation_count")) for doc in documents),
            "tables": sum(_as_int(doc.get("table_count")) for doc in documents),
            "formulas": sum(_as_int(doc.get("formula_count")) for doc in documents),
            "images": sum(_as_int(doc.get("image_count")) for doc in documents),
        },
        "chunks": {
            "total": len(chunks),
            "by_type": _counter(chunk_types),
            "by_source": _counter(chunk_sources),
            "empty_text": sum(1 for chunk in chunks if not chunk.text.strip()),
            "largest_by_chars_top20": chunk_lengths[:20],
        },
        "terms": {
            "total_entities": len(entities),
            "total_aliases": sum(len(entity.aliases) for entity in entities),
            "largest_alias_sets_top20": [
                {
                    "canonical": entity.canonical,
                    "alias_count": len(entity.aliases),
                    "aliases": entity.aliases[:25],
                }
                for entity in sorted(entities, key=lambda item: len(item.aliases), reverse=True)[:20]
            ],
        },
    }
    write_json(artifact_dir / STATS_REPORT_FILE, report)
    return report


def build_validation_report(artifact_dir: Path) -> dict[str, Any]:
    documents = load_records(artifact_dir / DOCUMENTS_FILE)
    chunks = load_chunks(artifact_dir / CHUNKS_FILE)
    entities = load_entities(artifact_dir / TERMS_FILE) if (artifact_dir / TERMS_FILE).exists() else []

    issues: list[ValidationIssue] = []
    issues.extend(validate_documents(documents))
    issues.extend(validate_chunks(chunks))
    issues.extend(validate_entities(entities))

    issue_records = [issue.to_record() for issue in issues]
    by_severity = Counter(issue.severity for issue in issues)
    by_kind = Counter(issue.kind for issue in issues)
    report = {
        "summary": {
            "issues": len(issues),
            "by_severity": _counter(by_severity),
            "by_kind": _counter(by_kind, limit=50),
        },
        "issues": issue_records,
    }
    write_json(artifact_dir / VALIDATION_REPORT_FILE, report)
    return report


def validate_documents(documents: list[dict[str, Any]]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    seen_ids: set[str] = set()
    for doc in documents:
        identifier = _document_id(doc)
        if identifier in seen_ids:
            issues.append(_issue("error", "duplicate_document_id", "Duplicate document identifier", "documents", identifier))
        seen_ids.add(identifier)

        for key in ["source", "title", "repo_rel_path", "osu_url"]:
            if not str(doc.get(key) or "").strip():
                issues.append(_issue("error", "missing_document_field", f"Missing `{key}`", "documents", identifier))

        sections = doc.get("sections")
        if not isinstance(sections, list) or not sections:
            issues.append(_issue("warning", "missing_sections", "Document has no sections", "documents", identifier))
        else:
            empty_sections = [
                str(section.get("section_id") or "")
                for section in sections
                if isinstance(section, dict) and not str(section.get("text") or "").strip()
            ]
            if len(empty_sections) == len(sections):
                issues.append(
                    _issue(
                        "info",
                        "only_empty_sections",
                        "Document has sections but none contain text",
                        "documents",
                        identifier,
                    )
                )
            elif len(empty_sections) > 20:
                issues.append(
                    _issue(
                        "info",
                        "many_empty_sections",
                        f"Document has {len(empty_sections)} empty layout sections",
                        "documents",
                        identifier,
                    )
                )

        if doc.get("source") == "osu-news":
            if not _looks_like_date(str(doc.get("date_iso") or "")):
                issues.append(_issue("warning", "invalid_news_date", "News post has invalid or missing date_iso", "documents", identifier))
            if not str(doc.get("preview_paragraph") or "").strip():
                issues.append(_issue("warning", "missing_news_preview", "News post has no preview paragraph", "documents", identifier))
            style_issues = doc.get("style_issues")
            if isinstance(style_issues, list) and style_issues:
                actionable_style_issues = [str(item) for item in style_issues if str(item) != "missing_banner_image"]
                if not actionable_style_issues:
                    continue
                issues.append(
                    _issue(
                        "info",
                        "news_style_issues",
                        "News post has style issues: " + ", ".join(actionable_style_issues[:8]),
                        "documents",
                        identifier,
                    )
                )
    return issues


def validate_chunks(chunks) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    seen_ids: set[str] = set()
    chunks_by_document: dict[str, int] = defaultdict(int)
    for chunk in chunks:
        chunks_by_document[chunk.document_id] += 1
        if chunk.id in seen_ids:
            issues.append(_issue("error", "duplicate_chunk_id", "Duplicate chunk identifier", "chunks", chunk.id))
        seen_ids.add(chunk.id)

        if not chunk.text.strip():
            issues.append(_issue("error", "empty_chunk_text", "Chunk text is empty", "chunks", chunk.id))
        if not chunk.osu_url.strip():
            issues.append(_issue("warning", "missing_chunk_url", "Chunk has no osu_url", "chunks", chunk.id))
        if not chunk.metadata.get("chunk_type"):
            issues.append(_issue("warning", "missing_chunk_type", "Chunk has no chunk_type metadata", "chunks", chunk.id))
        if len(chunk.text) > 9000:
            issues.append(_issue("warning", "large_chunk", f"Chunk has {len(chunk.text)} characters", "chunks", chunk.id))
        if chunk.metadata.get("chunk_type") == "table" and not chunk.metadata.get("table_headers"):
            issues.append(_issue("info", "table_missing_headers", "Table chunk has no table_headers metadata", "chunks", chunk.id))

    for document_id, count in chunks_by_document.items():
        if count > 500:
            issues.append(
                _issue("info", "many_chunks_per_document", f"Document produced {count} chunks", "chunks", document_id)
            )
    return issues


def validate_entities(entities) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    seen: set[str] = set()
    stopword_aliases = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "could",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "not",
        "of",
        "on",
        "or",
        "should",
        "the",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
    }
    for entity in entities:
        key = entity.canonical.casefold()
        if key in seen:
            issues.append(_issue("warning", "duplicate_entity", "Duplicate canonical entity", "terms", entity.canonical))
        seen.add(key)
        noisy = [alias for alias in entity.aliases if alias.casefold() in stopword_aliases]
        if noisy:
            issues.append(_issue("error", "stopword_alias", f"Entity has stopword aliases: {', '.join(noisy)}", "terms", entity.canonical))
        if len(entity.aliases) > 40:
            issues.append(_issue("info", "large_alias_set", f"Entity has {len(entity.aliases)} aliases", "terms", entity.canonical))
    return issues


def _counter(counter: Counter, limit: int | None = None) -> dict[str, int]:
    items = counter.most_common(limit)
    return {str(key): int(value) for key, value in items}


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _document_id(document: dict[str, Any]) -> str:
    return str(document.get("page_id") or document.get("post_id") or document.get("repo_rel_path") or "unknown")


def _looks_like_date(value: str) -> bool:
    return bool(re.match(r"^\d{4}-\d{2}-\d{2}T", value))


def _issue(severity: str, kind: str, message: str, artifact: str, identifier: str) -> ValidationIssue:
    return ValidationIssue(severity=severity, kind=kind, message=message, artifact=artifact, identifier=identifier)
