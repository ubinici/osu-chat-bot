from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

from ..domain.artifacts import read_jsonl, write_jsonl

FEEDBACK_LABELS = {"positive", "negative", "correction"}
REVIEW_DECISIONS = {"accept", "reject"}
TRAINING_SPLITS = {"train", "validation"}
DATASET_SPLITS = TRAINING_SPLITS | {"test"}


@dataclass(frozen=True)
class FeedbackEvent:
    """A source-neutral, append-only observation from a serving surface."""

    event_id: str
    occurred_at: str
    query: str
    feedback: str
    source: str
    analysis: dict[str, object] = field(default_factory=dict)
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    retrieved_document_ids: list[str] = field(default_factory=list)
    answer_version: str = "unknown"


@dataclass(frozen=True)
class FeedbackReview:
    """An offline human decision that may promote one feedback event."""

    event_id: str
    decision: str
    topic_ids: list[str] = field(default_factory=list)
    negative_topic_ids: list[str] = field(default_factory=list)
    split: str = "train"
    notes: str = ""


@dataclass(frozen=True)
class QueryTopicExample:
    """A reviewed query-to-topic label for a future classifier or reranker."""

    query: str
    topic_ids: list[str]
    split: str
    source: str
    source_event_id: str
    negative_topic_ids: list[str] = field(default_factory=list)


def load_feedback_events(path: Path) -> list[FeedbackEvent]:
    events: list[FeedbackEvent] = []
    for line_number, record in enumerate(read_jsonl(path), start=1):
        event = FeedbackEvent(
            event_id=_required_text(record, "event_id", path, line_number),
            occurred_at=_required_text(record, "occurred_at", path, line_number),
            query=_required_text(record, "query", path, line_number),
            feedback=_choice(record, "feedback", FEEDBACK_LABELS, path, line_number),
            source=_required_text(record, "source", path, line_number),
            analysis=_mapping(record.get("analysis"), "analysis", path, line_number),
            retrieved_chunk_ids=_text_list(record.get("retrieved_chunk_ids"), "retrieved_chunk_ids", path, line_number),
            retrieved_document_ids=_text_list(
                record.get("retrieved_document_ids"), "retrieved_document_ids", path, line_number
            ),
            answer_version=str(record.get("answer_version") or "unknown").strip() or "unknown",
        )
        events.append(event)
    _require_unique((event.event_id for event in events), "feedback event_id", path)
    return events


def load_feedback_reviews(path: Path) -> list[FeedbackReview]:
    reviews: list[FeedbackReview] = []
    for line_number, record in enumerate(read_jsonl(path), start=1):
        decision = _choice(record, "decision", REVIEW_DECISIONS, path, line_number)
        topic_ids = _text_list(record.get("topic_ids"), "topic_ids", path, line_number)
        split = str(record.get("split") or "train").strip().lower()
        if split not in TRAINING_SPLITS:
            raise ValueError(
                f"Invalid split {split!r} on line {line_number} in {path}; "
                f"expected one of {sorted(TRAINING_SPLITS)}"
            )
        if decision == "accept" and not topic_ids:
            raise ValueError(f"Accepted review needs topic_ids on line {line_number} in {path}")
        reviews.append(
            FeedbackReview(
                event_id=_required_text(record, "event_id", path, line_number),
                decision=decision,
                topic_ids=topic_ids,
                negative_topic_ids=_text_list(
                    record.get("negative_topic_ids"), "negative_topic_ids", path, line_number
                ),
                split=split,
                notes=str(record.get("notes") or "").strip(),
            )
        )
    _require_unique((review.event_id for review in reviews), "feedback review event_id", path)
    return reviews


def load_query_topic_dataset(path: Path) -> list[QueryTopicExample]:
    examples: list[QueryTopicExample] = []
    for line_number, record in enumerate(read_jsonl(path), start=1):
        split = _choice(record, "split", DATASET_SPLITS, path, line_number)
        topic_ids = _text_list(record.get("topic_ids"), "topic_ids", path, line_number)
        if not topic_ids:
            raise ValueError(f"Training example needs topic_ids on line {line_number} in {path}")
        examples.append(
            QueryTopicExample(
                query=_required_text(record, "query", path, line_number),
                topic_ids=topic_ids,
                split=split,
                source=_required_text(record, "source", path, line_number),
                source_event_id=_required_text(record, "source_event_id", path, line_number),
                negative_topic_ids=_text_list(
                    record.get("negative_topic_ids"), "negative_topic_ids", path, line_number
                ),
            )
        )
    _validate_query_examples(examples, path)
    return examples


def promote_feedback_dataset(
    feedback_path: Path,
    review_path: Path,
    output_path: Path,
    *,
    held_out_path: Path | None = None,
) -> dict[str, int]:
    """Promote explicitly reviewed feedback while protecting held-out queries."""

    events = load_feedback_events(feedback_path)
    reviews = load_feedback_reviews(review_path)
    event_by_id = {event.event_id: event for event in events}
    held_out_queries = _held_out_queries(held_out_path)
    examples: list[QueryTopicExample] = []
    rejected = 0
    excluded_held_out = 0

    for review in reviews:
        event = event_by_id.get(review.event_id)
        if event is None:
            raise ValueError(f"Review references unknown feedback event_id {review.event_id!r}")
        if review.decision == "reject":
            rejected += 1
            continue
        if normalize_query(event.query) in held_out_queries:
            excluded_held_out += 1
            continue
        examples.append(
            QueryTopicExample(
                query=event.query,
                topic_ids=review.topic_ids,
                negative_topic_ids=review.negative_topic_ids,
                split=review.split,
                source=f"reviewed_{event.source}_feedback",
                source_event_id=event.event_id,
            )
        )

    _validate_query_examples(examples, output_path)
    write_jsonl(output_path, examples)
    return {
        "feedback_events": len(events),
        "reviews": len(reviews),
        "promoted_examples": len(examples),
        "rejected_reviews": rejected,
        "excluded_held_out": excluded_held_out,
        "unreviewed_events": len(events) - len(reviews),
    }


def normalize_query(query: str) -> str:
    return " ".join(re.findall(r"\w+", query.casefold()))


def _held_out_queries(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {
        normalize_query(str(record.get("question") or record.get("query") or ""))
        for record in read_jsonl(path)
        if str(record.get("question") or record.get("query") or "").strip()
    }


def _validate_query_examples(examples: list[QueryTopicExample], path: Path) -> None:
    seen: dict[str, QueryTopicExample] = {}
    for example in examples:
        key = normalize_query(example.query)
        previous = seen.get(key)
        if previous is None:
            seen[key] = example
            continue
        if (
            previous.topic_ids != example.topic_ids
            or previous.negative_topic_ids != example.negative_topic_ids
            or previous.split != example.split
        ):
            raise ValueError(f"Conflicting labels or splits for duplicate query {example.query!r} in {path}")
        raise ValueError(f"Duplicate query {example.query!r} in {path}")


def _required_text(record: dict, field_name: str, path: Path, line_number: int) -> str:
    value = str(record.get(field_name) or "").strip()
    if not value:
        raise ValueError(f"Missing {field_name} on line {line_number} in {path}")
    return value


def _choice(
    record: dict,
    field_name: str,
    allowed: set[str],
    path: Path,
    line_number: int,
) -> str:
    value = _required_text(record, field_name, path, line_number).lower()
    if value not in allowed:
        raise ValueError(
            f"Invalid {field_name} {value!r} on line {line_number} in {path}; "
            f"expected one of {sorted(allowed)}"
        )
    return value


def _text_list(value: object, field_name: str, path: Path, line_number: int) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list on line {line_number} in {path}")
    items = [str(item).strip() for item in value]
    if any(not item for item in items):
        raise ValueError(f"{field_name} contains an empty value on line {line_number} in {path}")
    return list(dict.fromkeys(items))


def _mapping(value: object, field_name: str, path: Path, line_number: int) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object on line {line_number} in {path}")
    return value


def _require_unique(values, label: str, path: Path) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"Duplicate {label} {value!r} in {path}")
        seen.add(value)
