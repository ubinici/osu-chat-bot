from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import re
from typing import Any, Iterable

from ..domain.artifacts import (
    CANONICAL_TOPICS_FILE,
    CANONICAL_TOPICS_REPORT_FILE,
    DOCUMENTS_FILE,
    load_records,
    read_jsonl,
    write_json,
    write_jsonl,
)
from ..domain.source_profile import profile_for_document
from ..retrieval.lexical import document_id


TOPIC_ALIAS_RE = re.compile(r"[a-z0-9!+#.]+")
GENERIC_TOPIC_IDENTIFIERS = {
    "",
    "article",
    "details",
    "external links",
    "help",
    "links",
    "main article",
    "overview",
    "preamble",
    "references",
    "see also",
    "source",
    "the",
    "this article",
    "trivia",
    "wiki",
}

# Small, reviewable bridges for corpus IDs that existed in eval/stub history but
# now live under clearer current wiki page IDs. These are identifier aliases,
# not query phrase aliases.
COMPATIBLE_DOCUMENT_IDS = {
    "Beatmap/HP_drain_rate": ["Beatmap/Health_drain", "Beatmap/Health_drain_rate"],
    "Client/Interface/Chat_console": ["Chat_console"],
}
TOPIC_ID_OVERRIDES = {
    "Beatmap/HP_drain_rate": "Beatmap/Health_drain",
    "Client/Interface/Chat_console": "Chat_console",
}


def build_canonical_topic_artifacts(artifact_dir: Path, *, output_dir: Path | None = None) -> dict[str, Any]:
    output_dir = output_dir or artifact_dir
    documents = load_records(artifact_dir / DOCUMENTS_FILE)
    rows = build_canonical_topic_rows(documents)
    write_jsonl(output_dir / CANONICAL_TOPICS_FILE, rows)
    report = {
        "documents": len(documents),
        "topics": len(rows),
        "document_links": sum(len(row["document_ids"]) for row in rows),
        "equivalent_document_ids": sum(len(row["equivalent_document_ids"]) for row in rows),
        "aliases": sum(len(row["aliases"]) for row in rows),
        "by_source": dict(Counter(str(row["canonical_source"]) for row in rows).most_common()),
        "by_trust_tier": dict(Counter(str(row["trust_tier"]) for row in rows).most_common()),
        "by_retrieval_lane": dict(Counter(str(row["retrieval_lane"]) for row in rows).most_common()),
        "artifact": CANONICAL_TOPICS_FILE,
    }
    write_json(output_dir / CANONICAL_TOPICS_REPORT_FILE, report)
    return report


def load_or_build_canonical_topic_rows(artifact_dir: Path, documents: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = read_jsonl(artifact_dir / CANONICAL_TOPICS_FILE)
    if rows:
        return rows
    return build_canonical_topic_rows(list(documents))


def build_canonical_topic_rows(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    docs = {document_id(doc): doc for doc in documents if document_id(doc)}
    tail_counts = Counter(doc_id.rsplit("/", 1)[-1] for doc_id in docs)
    grouped: dict[str, list[str]] = defaultdict(list)
    for doc_id, document in docs.items():
        grouped[topic_id_for_document(doc_id, document)].append(doc_id)

    rows = []
    for topic_id, doc_ids in grouped.items():
        canonical_doc_id = choose_canonical_document_id(topic_id, doc_ids, docs)
        canonical_document = docs[canonical_doc_id]
        profile = profile_for_document(canonical_document)
        equivalent_ids = equivalent_document_ids(topic_id, canonical_doc_id, doc_ids, tail_counts)
        aliases = topic_aliases(canonical_document, canonical_doc_id, topic_id, equivalent_ids)
        rows.append(
            {
                "topic_id": topic_id,
                "canonical_document_id": canonical_doc_id,
                "title": str(canonical_document.get("title") or canonical_doc_id),
                "document_ids": sorted(set(doc_ids)),
                "equivalent_document_ids": equivalent_ids,
                "aliases": aliases,
                "alias_keys": [normalize_topic_alias(alias) for alias in aliases],
                "canonical_source": canonical_document.get("source"),
                "retrieval_lane": profile.lane,
                "trust_tier": profile.trust_tier,
                "confidence": 1.0,
            }
        )
    return sorted(rows, key=lambda row: (row["topic_id"], row["canonical_document_id"]))


def topic_id_for_document(doc_id: str, document: dict[str, Any]) -> str:
    explicit_topic_id = str(document.get("topic_id") or document.get("canonical_topic_id") or "").strip()
    if explicit_topic_id:
        return explicit_topic_id
    return TOPIC_ID_OVERRIDES.get(doc_id, doc_id)


def choose_canonical_document_id(topic_id: str, doc_ids: list[str], docs: dict[str, dict[str, Any]]) -> str:
    if topic_id in docs:
        return topic_id
    preferred = sorted(
        doc_ids,
        key=lambda doc_id: (
            0 if str(docs[doc_id].get("source") or "") == "osu-wiki" else 1,
            0 if profile_for_document(docs[doc_id]).trust_tier == "canonical" else 1,
            doc_id.count("/"),
            doc_id,
        ),
    )
    return preferred[0]


def equivalent_document_ids(
    topic_id: str,
    canonical_doc_id: str,
    doc_ids: Iterable[str],
    tail_counts: Counter[str],
) -> list[str]:
    values = [topic_id, canonical_doc_id, *doc_ids, *COMPATIBLE_DOCUMENT_IDS.get(canonical_doc_id, [])]
    for doc_id in doc_ids:
        tail = doc_id.rsplit("/", 1)[-1]
        if tail_counts[tail] == 1 and not is_generic_identifier(tail):
            values.append(tail)
    return sorted(unique_preserve(value for value in values if value), key=lambda value: (value.casefold(), value))


def topic_aliases(
    document: dict[str, Any],
    canonical_doc_id: str,
    topic_id: str,
    equivalent_ids: Iterable[str],
) -> list[str]:
    values = [
        str(document.get("title") or ""),
        display_identifier(canonical_doc_id),
        display_identifier(topic_id),
    ]
    values.extend(display_identifier(value) for value in equivalent_ids)
    values.extend(str(tag) for tag in document.get("tags", []) if isinstance(document.get("tags"), list))
    aliases = []
    seen_keys: set[str] = set()
    for value in values:
        key = normalize_topic_alias(value)
        if not key or key in GENERIC_TOPIC_IDENTIFIERS or key in seen_keys:
            continue
        seen_keys.add(key)
        aliases.append(value)
    return aliases


def topic_rows_by_document(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_document: dict[str, dict[str, Any]] = {}
    for row in rows:
        for doc_id in row.get("document_ids", []):
            if doc_id:
                by_document[str(doc_id)] = row
        canonical = str(row.get("canonical_document_id") or "")
        if canonical:
            by_document.setdefault(canonical, row)
    return by_document


def topic_rows_by_equivalent_id(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        for doc_id in row.get("equivalent_document_ids", []):
            if doc_id:
                by_id[str(doc_id)] = row
        topic_id = str(row.get("topic_id") or "")
        if topic_id:
            by_id.setdefault(topic_id, row)
    return by_id


def display_identifier(value: str) -> str:
    return value.replace("/", " ").replace("_", " ").strip()


def normalize_topic_alias(value: str) -> str:
    value = value.casefold()
    value = value.replace("_", " ").replace("-", " ").replace("/", " ")
    value = re.sub(r"[^a-z0-9!+#.]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def is_generic_identifier(value: str) -> bool:
    return normalize_topic_alias(value) in GENERIC_TOPIC_IDENTIFIERS


def unique_preserve(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result
