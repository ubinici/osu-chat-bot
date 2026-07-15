from __future__ import annotations

from collections import defaultdict

from .datasets import EvaluationExample
from ..domain.models import SearchResult


def score_retrieval(example: EvaluationExample, results: list[SearchResult]) -> dict[str, float | int]:
    expected_chunks = set(example.expected_chunk_ids)
    expected_documents = set(example.expected_document_ids)
    chunk_rank = _first_rank([result.chunk.id for result in results], expected_chunks)
    document_rank = _first_rank([result.chunk.document_id for result in results], expected_documents)
    chunk_match = chunk_rank > 0
    document_match = document_rank > 0
    has_expectation = bool(expected_chunks or expected_documents)
    matched = chunk_match or document_match if has_expectation else False
    matched_ranks = [rank for rank in (chunk_rank, document_rank) if rank > 0]
    matched_rank = min(matched_ranks, default=0)
    return {
        "has_expectation": int(has_expectation),
        "matched": int(matched),
        "chunk_match": int(chunk_match),
        "document_match": int(document_match),
        "matched_rank": matched_rank,
        "reciprocal_rank": 1.0 / matched_rank if matched_rank else 0.0,
        "hit_at_1": int(0 < matched_rank <= 1),
        "hit_at_3": int(0 < matched_rank <= 3),
        "hit_at_6": int(0 < matched_rank <= 6),
    }


def summarize_scores(scores: list[dict[str, float | int]]) -> dict[str, float | int]:
    total = len(scores)
    judged = sum(int(score["has_expectation"]) for score in scores)
    matched = sum(int(score["matched"]) for score in scores)
    hit_at_1 = sum(int(score.get("hit_at_1") or 0) for score in scores)
    hit_at_3 = sum(int(score.get("hit_at_3") or 0) for score in scores)
    hit_at_6 = sum(int(score.get("hit_at_6") or 0) for score in scores)
    reciprocal_rank_total = sum(float(score.get("reciprocal_rank") or 0.0) for score in scores)
    return {
        "examples": total,
        "judged_examples": judged,
        "matches": matched,
        "retrieval_accuracy": matched / judged if judged else 0.0,
        "hit_at_1": hit_at_1 / judged if judged else 0.0,
        "hit_at_3": hit_at_3 / judged if judged else 0.0,
        "hit_at_6": hit_at_6 / judged if judged else 0.0,
        "mean_reciprocal_rank": reciprocal_rank_total / judged if judged else 0.0,
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
        "matched_rank": int(row.get("matched_rank") or 0),
        "reciprocal_rank": float(row.get("reciprocal_rank") or 0.0),
        "hit_at_1": int(row.get("hit_at_1") or 0),
        "hit_at_3": int(row.get("hit_at_3") or 0),
        "hit_at_6": int(row.get("hit_at_6") or 0),
    }


def _first_rank(values: list[str], expected: set[str]) -> int:
    if not expected:
        return 0
    for index, value in enumerate(values, start=1):
        if value in expected:
            return index
    return 0
