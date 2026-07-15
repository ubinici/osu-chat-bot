from __future__ import annotations

from pathlib import Path
from typing import Any

from ..domain.artifacts import (
    DOCUMENT_ALIASES_FILE,
    DOCUMENTS_FILE,
    LINK_ALIAS_CANDIDATES_FILE,
    read_jsonl,
    write_jsonl,
)
from .links import term_key


def build_document_aliases(artifact_dir: Path, *, output_dir: Path | None = None) -> dict[str, int]:
    """Build a compact, source-neutral alias artifact for online resolution."""

    output_dir = output_dir or artifact_dir
    documents = read_jsonl(artifact_dir / DOCUMENTS_FILE)
    document_by_id = {
        document_id: record
        for record in documents
        if (document_id := _document_id(record))
    }
    rows: dict[tuple[str, str], dict[str, Any]] = {}

    for document_id, record in document_by_id.items():
        canonical_id = str(record.get("canonical_document_id") or document_id).strip()
        topic_document_ids = _topic_document_ids(record, canonical_id, document_id)
        candidates = [
            (record.get("title"), "title", 1.0),
            (_path_tail(document_id), "path_tail", 0.9),
        ]
        candidates.extend((alias, "source_alias", 0.95) for alias in _string_list(record.get("aliases")))
        for alias, source, confidence in candidates:
            _add_alias(
                rows,
                alias=alias,
                document_id=document_id,
                canonical_id=canonical_id,
                topic_document_ids=topic_document_ids,
                source=source,
                confidence=confidence,
                target_source=_target_source(record),
                retrieval_lane=_retrieval_lane(record),
            )

    accepted_links = 0
    for candidate in read_jsonl(artifact_dir / LINK_ALIAS_CANDIDATES_FILE):
        if candidate.get("decision") != "accept":
            continue
        document_id = str(candidate.get("target_page_id") or "").strip()
        record = document_by_id.get(document_id)
        if not document_id or record is None:
            continue
        accepted_links += 1
        canonical_id = str(record.get("canonical_document_id") or document_id).strip()
        _add_alias(
            rows,
            alias=candidate.get("alias"),
            document_id=document_id,
            canonical_id=canonical_id,
            topic_document_ids=_topic_document_ids(record, canonical_id, document_id),
            source="accepted_link",
            confidence=float(candidate.get("confidence") or 0.0),
            target_source=_target_source(record),
            retrieval_lane=_retrieval_lane(record),
        )

    ordered = sorted(rows.values(), key=lambda row: (row["alias_key"], row["document_id"]))
    write_jsonl(output_dir / DOCUMENT_ALIASES_FILE, ordered)
    return {
        "documents": len(document_by_id),
        "accepted_link_aliases": accepted_links,
        "aliases": len(ordered),
    }


def _add_alias(
    rows: dict[tuple[str, str], dict[str, Any]],
    *,
    alias: object,
    document_id: str,
    canonical_id: str,
    topic_document_ids: list[str],
    source: str,
    confidence: float,
    target_source: str,
    retrieval_lane: str,
) -> None:
    display = str(alias or "").replace("_", " ").strip()
    alias_key = term_key(display)
    if not alias_key or confidence <= 0:
        return
    key = (alias_key, document_id)
    row = {
        "alias": display,
        "alias_key": alias_key,
        "document_id": document_id,
        "canonical_document_id": canonical_id,
        "topic_id": canonical_id,
        "topic_document_ids": topic_document_ids,
        "source": source,
        "confidence": round(confidence, 6),
        "target_source": target_source,
        "retrieval_lane": retrieval_lane,
    }
    previous = rows.get(key)
    if previous is None or float(previous["confidence"]) < confidence:
        rows[key] = row


def _document_id(record: dict[str, Any]) -> str:
    for key in ("page_id", "post_id", "id", "repo_rel_path"):
        value = str(record.get(key) or "").strip()
        if value:
            return value
    return ""


def _topic_document_ids(record: dict[str, Any], canonical_id: str, document_id: str) -> list[str]:
    result = [canonical_id, document_id]
    for relation in record.get("relations", []):
        if not isinstance(relation, dict) or relation.get("type") not in {"alias_of", "equivalent", "parent"}:
            continue
        related_id = str(relation.get("document_id") or "").strip()
        if related_id:
            result.append(related_id)
    return list(dict.fromkeys(result))


def _target_source(record: dict[str, Any]) -> str:
    return str(record.get("source") or record.get("source_type") or "unknown").strip()


def _retrieval_lane(record: dict[str, Any]) -> str:
    explicit = str(record.get("retrieval_lane") or "").strip()
    if explicit:
        return explicit
    source_type = str(record.get("source_type") or "").strip()
    return "temporal" if source_type == "news" or record.get("source") == "osu-news" else "canonical"


def _path_tail(document_id: str) -> str:
    return document_id.rstrip("/").rsplit("/", 1)[-1].replace("_", " ")


def _string_list(value: object) -> list[str]:
    return [str(item).strip() for item in value if str(item).strip()] if isinstance(value, list) else []
