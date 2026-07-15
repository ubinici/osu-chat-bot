from __future__ import annotations

from collections import defaultdict

from .datasets import EvaluationExample
from ..domain.models import SearchResult


def score_retrieval(example: EvaluationExample, results: list[SearchResult]) -> dict[str, object]:
    expected_chunks = set(example.expected_chunk_ids)
    expected_documents = set(example.expected_document_ids)
    acceptable_documents = set(example.acceptable_document_ids) - expected_documents
    chunk_rank, matched_chunk_id = _first_match(
        [result.chunk.id for result in results], expected_chunks
    )
    primary_document_rank, primary_document_id = _first_match(
        [result.chunk.document_id for result in results], expected_documents
    )
    acceptable_document_rank, acceptable_document_id = _first_match(
        [result.chunk.document_id for result in results], acceptable_documents
    )
    document_rank, matched_document_id, document_match_kind = _document_match(
        primary_document_rank,
        primary_document_id,
        acceptable_document_rank,
        acceptable_document_id,
    )
    chunk_match = chunk_rank > 0
    document_match = document_rank > 0
    primary_document_match = primary_document_rank > 0
    acceptable_document_match = acceptable_document_rank > 0
    has_expectation = bool(expected_chunks or expected_documents or acceptable_documents)
    strict_matched = chunk_match or primary_document_match if has_expectation else False
    matched = chunk_match or document_match if has_expectation else False
    matched_ranks = [rank for rank in (chunk_rank, document_rank) if rank > 0]
    matched_rank = min(matched_ranks, default=0)
    strict_ranks = [rank for rank in (chunk_rank, primary_document_rank) if rank > 0]
    strict_matched_rank = min(strict_ranks, default=0)
    return {
        "has_expectation": int(has_expectation),
        "matched": int(matched),
        "strict_matched": int(strict_matched),
        "acceptable_only_match": int(matched and not strict_matched),
        "chunk_match": int(chunk_match),
        "matched_chunk_id": matched_chunk_id,
        "document_match": int(document_match),
        "primary_document_match": int(primary_document_match),
        "acceptable_document_match": int(acceptable_document_match),
        "matched_document_id": matched_document_id,
        "document_match_kind": document_match_kind,
        "matched_rank": matched_rank,
        "reciprocal_rank": 1.0 / matched_rank if matched_rank else 0.0,
        "hit_at_1": int(0 < matched_rank <= 1),
        "hit_at_3": int(0 < matched_rank <= 3),
        "hit_at_6": int(0 < matched_rank <= 6),
        "strict_matched_rank": strict_matched_rank,
        "strict_reciprocal_rank": 1.0 / strict_matched_rank if strict_matched_rank else 0.0,
        "strict_hit_at_1": int(0 < strict_matched_rank <= 1),
        "strict_hit_at_3": int(0 < strict_matched_rank <= 3),
        "strict_hit_at_6": int(0 < strict_matched_rank <= 6),
    }


def summarize_scores(scores: list[dict[str, object]]) -> dict[str, float | int]:
    total = len(scores)
    judged = sum(int(score["has_expectation"]) for score in scores)
    matched = sum(int(score["matched"]) for score in scores)
    strict_matched = sum(int(score.get("strict_matched") or 0) for score in scores)
    acceptable_only_matches = sum(
        int(score.get("acceptable_only_match") or 0) for score in scores
    )
    hit_at_1 = sum(int(score.get("hit_at_1") or 0) for score in scores)
    hit_at_3 = sum(int(score.get("hit_at_3") or 0) for score in scores)
    hit_at_6 = sum(int(score.get("hit_at_6") or 0) for score in scores)
    reciprocal_rank_total = sum(float(score.get("reciprocal_rank") or 0.0) for score in scores)
    strict_hit_at_1 = sum(int(score.get("strict_hit_at_1") or 0) for score in scores)
    strict_hit_at_3 = sum(int(score.get("strict_hit_at_3") or 0) for score in scores)
    strict_hit_at_6 = sum(int(score.get("strict_hit_at_6") or 0) for score in scores)
    strict_reciprocal_rank_total = sum(
        float(score.get("strict_reciprocal_rank") or 0.0) for score in scores
    )
    return {
        "examples": total,
        "judged_examples": judged,
        "matches": matched,
        "strict_matches": strict_matched,
        "acceptable_only_matches": acceptable_only_matches,
        "retrieval_accuracy": matched / judged if judged else 0.0,
        "strict_retrieval_accuracy": strict_matched / judged if judged else 0.0,
        "hit_at_1": hit_at_1 / judged if judged else 0.0,
        "hit_at_3": hit_at_3 / judged if judged else 0.0,
        "hit_at_6": hit_at_6 / judged if judged else 0.0,
        "mean_reciprocal_rank": reciprocal_rank_total / judged if judged else 0.0,
        "strict_hit_at_1": strict_hit_at_1 / judged if judged else 0.0,
        "strict_hit_at_3": strict_hit_at_3 / judged if judged else 0.0,
        "strict_hit_at_6": strict_hit_at_6 / judged if judged else 0.0,
        "strict_mean_reciprocal_rank": (
            strict_reciprocal_rank_total / judged if judged else 0.0
        ),
    }


def summarize_scores_by_category(scores: list[dict[str, object]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for score in scores:
        grouped[str(score.get("category") or "uncategorized")].append(score)
    return {
        category: summarize_scores([narrow_score(row) for row in rows])
        for category, rows in sorted(grouped.items())
    }


def narrow_score(row: dict[str, object]) -> dict[str, object]:
    return {
        "has_expectation": int(row.get("has_expectation") or 0),
        "matched": int(row.get("matched") or 0),
        "strict_matched": int(row.get("strict_matched") or 0),
        "acceptable_only_match": int(row.get("acceptable_only_match") or 0),
        "chunk_match": int(row.get("chunk_match") or 0),
        "document_match": int(row.get("document_match") or 0),
        "matched_rank": int(row.get("matched_rank") or 0),
        "reciprocal_rank": float(row.get("reciprocal_rank") or 0.0),
        "hit_at_1": int(row.get("hit_at_1") or 0),
        "hit_at_3": int(row.get("hit_at_3") or 0),
        "hit_at_6": int(row.get("hit_at_6") or 0),
        "strict_matched_rank": int(row.get("strict_matched_rank") or 0),
        "strict_reciprocal_rank": float(row.get("strict_reciprocal_rank") or 0.0),
        "strict_hit_at_1": int(row.get("strict_hit_at_1") or 0),
        "strict_hit_at_3": int(row.get("strict_hit_at_3") or 0),
        "strict_hit_at_6": int(row.get("strict_hit_at_6") or 0),
    }


def _first_match(values: list[str], expected: set[str]) -> tuple[int, str | None]:
    if not expected:
        return 0, None
    for index, value in enumerate(values, start=1):
        if value in expected:
            return index, value
    return 0, None


def _document_match(
    primary_rank: int,
    primary_id: str | None,
    acceptable_rank: int,
    acceptable_id: str | None,
) -> tuple[int, str | None, str]:
    if primary_rank and (not acceptable_rank or primary_rank <= acceptable_rank):
        return primary_rank, primary_id, "primary"
    if acceptable_rank:
        return acceptable_rank, acceptable_id, "acceptable"
    return 0, None, "none"
