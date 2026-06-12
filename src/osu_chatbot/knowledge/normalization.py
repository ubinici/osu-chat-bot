from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv
import re

from ..domain.artifacts import (
    DOCUMENTS_FILE,
    ENTITY_CANDIDATES_FILE,
    ENTITY_NORMALIZATION_FILE,
    ENTITY_NORMALIZATION_REPORT_FILE,
    ENTITY_NORMALIZATION_REVIEW_FILE,
    load_records,
    read_jsonl,
    write_json,
    write_jsonl,
)
from .terms import acronym_for, normalize_term

MANUAL_ALIAS_TARGETS = {
    "ar": "Beatmap/Approach_rate",
    "bss": "Beatmapping/Beatmap_submission",
    "bancho": "BanchoBot",
    "grid snap": "Beatmapping/Grid_snapping",
}

GENERIC_NORMALIZED_TERMS = {
    "beatmaps",
    "beatmapping",
    "gameplay",
    "mappers",
}


@dataclass(frozen=True)
class DocumentIndexEntry:
    page_id: str
    title: str
    source: str
    repo_rel_path: str


def build_entity_normalization_artifacts(
    artifact_dir: Path,
    *,
    source_artifact_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    source_artifact_dir = source_artifact_dir or artifact_dir
    output_dir = output_dir or artifact_dir
    documents = load_records(source_artifact_dir / DOCUMENTS_FILE)
    candidates = read_jsonl(artifact_dir / ENTITY_CANDIDATES_FILE)
    document_index = build_document_index(documents)
    normalized = normalize_entity_candidates(candidates, document_index)

    write_jsonl(output_dir / ENTITY_NORMALIZATION_FILE, normalized)
    write_normalization_review_csv(output_dir / ENTITY_NORMALIZATION_REVIEW_FILE, normalized)

    report = {
        "candidates": len(candidates),
        "normalized_rows": len(normalized),
        "by_decision": dict(Counter(row["decision"] for row in normalized).most_common()),
        "by_label": dict(Counter(row["label"] for row in normalized).most_common()),
        "linked_rows": sum(1 for row in normalized if row.get("target_page_id")),
        "unique_canonical_entities": len({row["canonical"] for row in normalized if row["decision"] != "reject"}),
        "normalization_artifact": ENTITY_NORMALIZATION_FILE,
        "review_artifact": ENTITY_NORMALIZATION_REVIEW_FILE,
    }
    write_json(output_dir / ENTITY_NORMALIZATION_REPORT_FILE, report)
    return report


def normalize_entity_candidates(
    candidates: list[dict[str, Any]],
    document_index: dict[str, list[DocumentIndexEntry]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for candidate in candidates:
        row = normalize_candidate(candidate, document_index)
        key = (
            row["canonical_key"],
            row["label"],
            row.get("target_page_id") or row["source_document_id"],
        )
        previous = grouped.get(key)
        if previous is None or row["confidence"] > previous["confidence"]:
            grouped[key] = row
            continue
        previous["aliases"] = sorted(set(previous["aliases"]) | set(row["aliases"]), key=str.lower)
        previous["mentions"] += row["mentions"]
        previous["source_document_ids"] = sorted(
            set(previous["source_document_ids"]) | set(row["source_document_ids"]),
            key=str.lower,
        )

    return sorted(
        grouped.values(),
        key=lambda row: (decision_sort(row["decision"]), -row["confidence"], row["canonical"].casefold()),
    )


def normalize_candidate(candidate: dict[str, Any], document_index: dict[str, list[DocumentIndexEntry]]) -> dict[str, Any]:
    text = normalize_term(str(candidate.get("text") or ""))
    label = str(candidate.get("label") or "")
    source_document_id = str(candidate.get("document_id") or "")
    confidence = float(candidate.get("score") or 0.0)
    source_document = find_document_by_page_id(document_index, source_document_id)
    target, link_reason = resolve_target_document(text, source_document_id, source_document, document_index)
    canonical = target.title if target else normalize_canonical_text(text)
    decision, decision_reason = decision_for(text, confidence, target, link_reason)
    aliases = sorted({text, normalize_canonical_text(text), canonical}, key=str.lower)

    return {
        "decision": decision,
        "decision_reason": decision_reason,
        "canonical": canonical,
        "canonical_key": entity_key(canonical),
        "aliases": [alias for alias in aliases if alias],
        "label": label,
        "confidence": round(confidence, 3),
        "target_page_id": target.page_id if target else "",
        "target_title": target.title if target else "",
        "link_reason": link_reason,
        "source_document_id": source_document_id,
        "source_document_ids": [source_document_id] if source_document_id else [],
        "source_chunk_id": str(candidate.get("chunk_id") or ""),
        "mentions": 1,
        "example_text": text,
        "example_context": str(candidate.get("context") or ""),
    }


def build_document_index(documents: list[dict[str, Any]]) -> dict[str, list[DocumentIndexEntry]]:
    index: dict[str, list[DocumentIndexEntry]] = defaultdict(list)
    for document in documents:
        page_id = str(document.get("page_id") or "")
        if not page_id:
            continue
        entry = DocumentIndexEntry(
            page_id=page_id,
            title=str(document.get("title") or page_id.rsplit("/", 1)[-1].replace("_", " ")),
            source=str(document.get("source") or ""),
            repo_rel_path=str(document.get("repo_rel_path") or ""),
        )
        for key in document_keys(entry):
            index[key].append(entry)
    return index


def resolve_target_document(
    text: str,
    source_document_id: str,
    source_document: DocumentIndexEntry | None,
    document_index: dict[str, list[DocumentIndexEntry]],
) -> tuple[DocumentIndexEntry | None, str]:
    manual_target = MANUAL_ALIAS_TARGETS.get(entity_key(text))
    if manual_target:
        target = find_document_by_page_id(document_index, manual_target)
        if target:
            return target, "manual_alias"

    text_key = entity_key(text)
    source_title_key = entity_key(source_document.title) if source_document else ""
    if source_document and keys_match(text_key, source_title_key):
        return source_document, "source_title"

    for lookup_key in [text_key, singular_key(text_key)]:
        matches = document_index.get(lookup_key, [])
        if len(matches) == 1:
            return matches[0], "global_title"
        if len(matches) > 1:
            source_match = next((match for match in matches if match.page_id == source_document_id), None)
            if source_match:
                return source_match, "ambiguous_source_title"
            return matches[0], "ambiguous_global_title"

    if source_document and text_key in {entity_key(source_document.page_id.rsplit("/", 1)[-1]), singular_key(source_title_key)}:
        return source_document, "source_path_tail"
    return None, "unlinked"


def document_keys(entry: DocumentIndexEntry) -> set[str]:
    title = normalize_canonical_text(entry.title)
    tail = entry.page_id.rsplit("/", 1)[-1].replace("_", " ")
    keys = {
        entity_key(title),
        singular_key(entity_key(title)),
        entity_key(tail),
        singular_key(entity_key(tail)),
        entity_key(entry.page_id.replace("/", " ")),
    }
    acronym = acronym_for(title)
    if acronym:
        keys.add(entity_key(acronym))
    return {key for key in keys if key}


def find_document_by_page_id(
    document_index: dict[str, list[DocumentIndexEntry]],
    page_id: str,
) -> DocumentIndexEntry | None:
    for entries in document_index.values():
        for entry in entries:
            if entry.page_id == page_id:
                return entry
    return None


def decision_for(
    text: str,
    confidence: float,
    target: DocumentIndexEntry | None,
    link_reason: str,
) -> tuple[str, str]:
    key = entity_key(text)
    if key in GENERIC_NORMALIZED_TERMS:
        return "reject", "generic_domain_word"
    if target and confidence >= 0.7:
        return "accept", f"linked_{link_reason}"
    if target:
        return "review", f"linked_{link_reason}_low_confidence"
    if confidence >= 0.75 and len(text) >= 4:
        return "review", "high_confidence_unlinked"
    return "review", "unlinked_low_confidence"


def write_normalization_review_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "decision",
                "decision_reason",
                "confidence",
                "canonical",
                "aliases",
                "label",
                "target_page_id",
                "source_document_id",
                "mentions",
                "example_context",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "decision": row["decision"],
                    "decision_reason": row["decision_reason"],
                    "confidence": row["confidence"],
                    "canonical": row["canonical"],
                    "aliases": " | ".join(row["aliases"]),
                    "label": row["label"],
                    "target_page_id": row["target_page_id"],
                    "source_document_id": row["source_document_id"],
                    "mentions": row["mentions"],
                    "example_context": row["example_context"],
                }
            )
    temp_path.replace(path)


def normalize_canonical_text(value: str) -> str:
    value = normalize_term(value)
    return value[:1].upper() + value[1:] if value and value.islower() else value


def entity_key(value: str) -> str:
    value = normalize_term(value).casefold()
    value = re.sub(r"[^a-z0-9!+ ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def singular_key(value: str) -> str:
    if value.endswith("ies") and len(value) > 4:
        return value[:-3] + "y"
    if value.endswith("s") and len(value) > 4 and not value.endswith("ss"):
        return value[:-1]
    return value


def keys_match(left: str, right: str) -> bool:
    return left == right or singular_key(left) == singular_key(right)


def decision_sort(value: str) -> int:
    return {"accept": 0, "review": 1, "reject": 2}.get(value, 9)
