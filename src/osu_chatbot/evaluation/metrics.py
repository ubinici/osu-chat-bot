from __future__ import annotations

from collections import defaultdict

from .datasets import EvaluationExample
from ..domain.models import SearchResult


def score_retrieval(example: EvaluationExample, results: list[SearchResult]) -> dict[str, float | int]:
    chunk_hits = {result.chunk.id for result in results}
    document_hits = {result.chunk.document_id for result in results}
    topic_hits = {result.topic_id for result in results if result.topic_id}
    topic_document_hits = {
        topic_document_id
        for result in results
        for topic_document_id in result.topic_document_ids
        if topic_document_id
    }
    expected_chunks = set(example.expected_chunk_ids)
    expected_documents = set(example.expected_document_ids)
    chunk_match = bool(expected_chunks and expected_chunks.intersection(chunk_hits))
    topic_match = bool(expected_documents and expected_documents.intersection(topic_hits | topic_document_hits))
    document_match = bool(expected_documents and expected_documents.intersection(document_hits | topic_hits | topic_document_hits))
    has_expectation = bool(expected_chunks or expected_documents)
    matched = chunk_match or document_match if has_expectation else False
    return {
        "has_expectation": int(has_expectation),
        "matched": int(matched),
        "chunk_match": int(chunk_match),
        "document_match": int(document_match),
        "topic_match": int(topic_match),
    }


def summarize_scores(scores: list[dict[str, float | int]]) -> dict[str, float | int]:
    total = len(scores)
    judged = sum(int(score["has_expectation"]) for score in scores)
    matched = sum(int(score["matched"]) for score in scores)
    return {
        "examples": total,
        "judged_examples": judged,
        "matches": matched,
        "retrieval_accuracy": matched / judged if judged else 0.0,
    }


def summarize_scores_by_category(scores: list[dict[str, object]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for score in scores:
        grouped[str(score.get("category") or "uncategorized")].append(score)
    return {
        category: summarize_scores([narrow_score(row) for row in rows])
        for category, rows in sorted(grouped.items())
    }


def narrow_score(row: dict[str, object]) -> dict[str, float | int]:
    return {
        "has_expectation": int(row.get("has_expectation") or 0),
        "matched": int(row.get("matched") or 0),
        "chunk_match": int(row.get("chunk_match") or 0),
        "document_match": int(row.get("document_match") or 0),
        "topic_match": int(row.get("topic_match") or 0),
    }
