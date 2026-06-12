from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import csv
import re

from ..corpus.markdown import strip_markdown
from ..domain.artifacts import (
    DOCUMENTS_FILE,
    LINK_ALIAS_CANDIDATES_FILE,
    LINK_ALIAS_REVIEW_FILE,
    LINKS_RAW_FILE,
    LINKS_REPORT_FILE,
    load_records,
    write_json,
    write_jsonl,
)

GENERIC_ANCHORS = {
    "",
    "a",
    "an",
    "and",
    "article",
    "click here",
    "details",
    "here",
    "link",
    "main article",
    "more",
    "page",
    "read more",
    "see",
    "source",
    "the",
    "this",
    "this article",
    "this page",
    "website",
    "wiki page",
}


def build_link_artifacts(artifact_dir: Path) -> dict[str, Any]:
    documents = load_records(artifact_dir / DOCUMENTS_FILE)
    rows = collect_link_rows(documents)
    wiki_page_ids = {str(doc.get("page_id")) for doc in documents if doc.get("source") == "osu-wiki" and doc.get("page_id")}
    candidates = build_link_candidates(rows, wiki_page_ids=wiki_page_ids)
    write_jsonl(artifact_dir / LINKS_RAW_FILE, rows)
    write_jsonl(artifact_dir / LINK_ALIAS_CANDIDATES_FILE, candidates)
    write_review_csv(artifact_dir / "link_alias_candidates.csv", candidates)
    write_review_csv(artifact_dir / "link_alias_accept.csv", [item for item in candidates if item["decision"] == "accept"])
    write_review_csv(artifact_dir / LINK_ALIAS_REVIEW_FILE, [item for item in candidates if item["decision"] == "review"])
    write_review_csv(artifact_dir / "link_alias_reject.csv", [item for item in candidates if item["decision"] == "reject"])

    report = {
        "documents": len(documents),
        "raw_links": len(rows),
        "internal_wiki_links": sum(1 for row in rows if row["is_internal_wiki"]),
        "known_wiki_pages": len(wiki_page_ids),
        "candidate_aliases": len(candidates),
        "accepted_candidates": sum(1 for item in candidates if item["decision"] == "accept"),
        "review_candidates": sum(1 for item in candidates if item["decision"] == "review"),
        "rejected_candidates": sum(1 for item in candidates if item["decision"] == "reject"),
        "top_link_types": dict(Counter(row["link_type"] for row in rows).most_common(20)),
    }
    write_json(artifact_dir / LINKS_REPORT_FILE, report)
    return report


def collect_link_rows(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for document in documents:
        source_id = str(document.get("page_id") or document.get("post_id") or document.get("repo_rel_path") or "")
        for edge in _list_of_dicts(document.get("link_edges")):
            link_text = normalize_anchor_text(str(edge.get("link_text") or ""))
            target_page_id = normalize_target_page_id(str(edge.get("target_page_id") or ""))
            link_type = str(edge.get("link_type") or "")
            rows.append(
                {
                    "source": document.get("source"),
                    "source_id": source_id,
                    "source_title": document.get("title"),
                    "source_repo_rel_path": document.get("repo_rel_path"),
                    "source_section_id": edge.get("source_section_id"),
                    "source_heading_path": edge.get("source_heading_path") or [],
                    "link_text": link_text,
                    "link_text_key": term_key(link_text),
                    "url_raw": edge.get("url_raw"),
                    "url_resolved": edge.get("url_resolved"),
                    "link_type": link_type,
                    "target_page_id": target_page_id,
                    "target_anchor": edge.get("target_anchor"),
                    "context": edge.get("context") or "",
                    "is_internal_wiki": link_type in {"wiki_article", "wiki_section"} and bool(target_page_id),
                }
            )
    return rows


def build_link_candidates(rows: list[dict[str, Any]], wiki_page_ids: set[str] | None = None) -> list[dict[str, Any]]:
    wiki_page_ids = wiki_page_ids or set()
    by_alias: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["is_internal_wiki"] and row["link_text_key"] and target_is_entity_candidate(str(row["target_page_id"]), wiki_page_ids):
            by_alias[row["link_text_key"]].append(row)

    candidates: list[dict[str, Any]] = []
    for alias_key, alias_rows in sorted(by_alias.items()):
        target_counts = Counter(str(row["target_page_id"]) for row in alias_rows if row.get("target_page_id"))
        if not target_counts:
            continue
        target_page_id, target_count = target_counts.most_common(1)[0]
        alias_text = most_common_display(alias_rows)
        ambiguity = len(target_counts)
        source_count = len({str(row["source_id"]) for row in alias_rows})
        confidence = score_candidate(alias_text, target_page_id, target_count, len(alias_rows), source_count, ambiguity)
        decision = decision_for(alias_text, target_page_id, confidence, source_count, len(alias_rows), ambiguity)
        decision_reason = reason_for(alias_text, target_page_id, confidence, source_count, len(alias_rows), ambiguity, decision)
        candidates.append(
            {
                "alias": alias_text,
                "alias_key": alias_key,
                "target_page_id": target_page_id,
                "target_title_guess": target_page_id.rsplit("/", 1)[-1].replace("_", " "),
                "decision": decision,
                "decision_reason": decision_reason,
                "confidence": round(confidence, 3),
                "source_count": source_count,
                "link_count": len(alias_rows),
                "target_count": target_count,
                "ambiguous_target_count": ambiguity,
                "other_targets": [
                    {"target_page_id": target, "count": count}
                    for target, count in target_counts.most_common()[1:6]
                ],
                "examples": [
                    {
                        "source_id": row["source_id"],
                        "source_title": row["source_title"],
                        "heading_path": row["source_heading_path"],
                        "context": row.get("context", ""),
                    }
                    for row in alias_rows[:5]
                ],
            }
        )
    return sorted(candidates, key=candidate_sort_key)


def write_review_csv(path: Path, candidates: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "decision",
                "decision_reason",
                "confidence",
                "alias",
                "target_page_id",
                "target_title_guess",
                "source_count",
                "link_count",
                "ambiguous_target_count",
                "other_targets",
                "example_context",
            ],
        )
        writer.writeheader()
        for item in candidates:
            writer.writerow(
                {
                    "decision": item["decision"],
                    "decision_reason": item["decision_reason"],
                    "confidence": item["confidence"],
                    "alias": item["alias"],
                    "target_page_id": item["target_page_id"],
                    "target_title_guess": item["target_title_guess"],
                    "source_count": item["source_count"],
                    "link_count": item["link_count"],
                    "ambiguous_target_count": item["ambiguous_target_count"],
                    "other_targets": "; ".join(
                        f"{target['target_page_id']} ({target['count']})" for target in item["other_targets"]
                    ),
                    "example_context": first_example_context(item),
                }
            )
    temp_path.replace(path)


def score_candidate(alias: str, target_page_id: str, target_count: int, total_count: int, source_count: int, ambiguity: int) -> float:
    alias_key = term_key(alias)
    target_tail_key = term_key(target_page_id.rsplit("/", 1)[-1].replace("_", " "))
    target_full_key = term_key(target_page_id.replace("/", " ").replace("_", " "))
    score = 0.25
    score += min(0.22, target_count / 25)
    score += min(0.18, source_count / 20)
    score += 0.25 * (target_count / total_count)
    if alias_key == target_tail_key:
        score += 0.25
    elif alias_key and (alias_key in target_full_key or target_tail_key in alias_key):
        score += 0.12
    if ambiguity > 1:
        score -= min(0.35, 0.08 * (ambiguity - 1))
    if is_generic_anchor(alias):
        score -= 0.6
    if len(alias_key) < 3:
        score -= 0.3
    if is_tournament_edition_anchor(alias, target_page_id):
        score -= 0.25
    return max(0.0, min(1.0, score))


def decision_for(alias: str, target_page_id: str, confidence: float, source_count: int, link_count: int, ambiguity: int) -> str:
    if is_generic_anchor(alias) or confidence < 0.35:
        return "reject"
    if is_tournament_edition_anchor(alias, target_page_id):
        return "review"
    if confidence >= 0.78 and ambiguity <= 2 and (source_count >= 2 or link_count >= 4):
        return "accept"
    return "review"


def reason_for(alias: str, target_page_id: str, confidence: float, source_count: int, link_count: int, ambiguity: int, decision: str) -> str:
    if is_generic_anchor(alias):
        return "generic_anchor"
    if confidence < 0.35:
        return "low_confidence"
    if is_tournament_edition_anchor(alias, target_page_id):
        return "tournament_or_year_specific"
    if ambiguity > 2:
        return "ambiguous_targets"
    if decision == "accept":
        return "repeated_high_confidence"
    if source_count < 2 and link_count < 4:
        return "weak_single_source_evidence"
    if ambiguity > 1:
        return "minor_target_ambiguity"
    return "needs_manual_confirmation"


def normalize_anchor_text(value: str) -> str:
    value = strip_markdown(value or "")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def normalize_target_page_id(value: str) -> str:
    value = value.strip().strip("/")
    if value.startswith("wiki/"):
        value = value[len("wiki/") :]
    return value


def term_key(value: str) -> str:
    value = normalize_anchor_text(value).replace("_", " ").replace("-", " ")
    value = re.sub(r"\s+", " ", value).strip().casefold()
    return value


def is_generic_anchor(value: str) -> bool:
    key = term_key(value)
    return (
        key in GENERIC_ANCHORS
        or bool(re.fullmatch(r"(read|see|learn) (this|more|here|article|page)", key))
        or bool(re.fullmatch(r".{0,40}\bwiki page", key))
        or bool(re.fullmatch(r"\d{4}", key))
    )


def target_is_entity_candidate(target_page_id: str, wiki_page_ids: set[str]) -> bool:
    if not target_page_id or target_page_id.startswith("news/"):
        return False
    if wiki_page_ids and target_page_id not in wiki_page_ids:
        return False
    return True


def is_tournament_edition_anchor(alias: str, target_page_id: str) -> bool:
    key = term_key(alias)
    if not target_page_id.startswith("Tournaments/") and not target_page_id.startswith("Contests/"):
        return False
    return bool(re.fullmatch(r"\d{4}( (4k|7k))?", key)) or bool(re.search(r"\b(20\d{2}|19\d{2})\b", key))


def most_common_display(rows: list[dict[str, Any]]) -> str:
    counts = Counter(str(row["link_text"]) for row in rows if str(row.get("link_text") or "").strip())
    if not counts:
        return ""
    return counts.most_common(1)[0][0]


def decision_sort(value: str) -> int:
    return {"accept": 0, "review": 1, "reject": 2}.get(value, 9)


def candidate_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    decision = item["decision"]
    if decision == "review":
        return (
            decision_sort(decision),
            item["decision_reason"],
            -int(item["source_count"]),
            -int(item["link_count"]),
            -float(item["confidence"]),
            item["alias_key"],
        )
    return (decision_sort(decision), -float(item["confidence"]), item["alias_key"])


def _list_of_dicts(value: object) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def first_example_context(item: dict[str, Any]) -> str:
    for example in item.get("examples", []):
        context = str(example.get("context") or "").strip()
        if context:
            return context
    return ""
