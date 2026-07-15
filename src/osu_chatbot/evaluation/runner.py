from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..retrieval.service import Retriever
from .datasets import load_evaluation_dataset
from .metrics import score_retrieval, summarize_scores, summarize_scores_by_category


def run_evaluation(
    config: AppConfig,
    dataset_path: Path,
    *,
    retriever: Retriever | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    examples = load_evaluation_dataset(dataset_path)
    retriever = retriever or Retriever(config)
    rows = []
    for example in examples:
        outcome = retriever.retrieve(example.question, top_k=top_k)
        results = outcome.results
        score = score_retrieval(example, results)
        rows.append(
            {
                "question": example.question,
                "category": example.category,
                "expected_chunk_ids": example.expected_chunk_ids,
                "expected_document_ids": example.expected_document_ids,
                "acceptable_document_ids": example.acceptable_document_ids,
                "intent": sorted(outcome.intent.labels),
                "retrieval_lane": outcome.analysis.retrieval_lane,
                "resolved_topics": [
                    {
                        "canonical_id": topic.canonical_id,
                        "matched_alias": topic.matched_alias,
                        "document_ids": list(topic.document_ids),
                        "confidence": round(topic.confidence, 6),
                        "preference_strength": topic.preference_strength,
                    }
                    for topic in outcome.analysis.topics
                ],
                "search_query": outcome.search_query,
                "top_chunk_ids": [result.chunk.id for result in results],
                "top_document_ids": [result.chunk.document_id for result in results],
                "top_sources": [
                    {
                        "chunk_id": result.chunk.id,
                        "document_id": result.chunk.document_id,
                        "title": result.chunk.title,
                        "score": round(result.score, 6),
                    }
                    for result in results
                ],
                **score,
            }
        )
    return {"summary": summarize_scores(rows), "by_category": summarize_scores_by_category(rows), "examples": rows}
