from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..retrieval.service import Retriever
from .datasets import load_evaluation_dataset
from .metrics import score_retrieval, summarize_scores, summarize_scores_by_category


def run_evaluation(config: AppConfig, dataset_path: Path, *, use_dense: bool = False) -> dict[str, Any]:
    examples = load_evaluation_dataset(dataset_path)
    retriever = Retriever(config, use_dense=use_dense)
    rows = []
    for example in examples:
        entities, results = retriever.search(example.question)
        score = score_retrieval(example, results)
        rows.append(
            {
                "question": example.question,
                "category": example.category,
                "expected_chunk_ids": example.expected_chunk_ids,
                "expected_document_ids": example.expected_document_ids,
                "entities": [entity.canonical for entity in entities],
                "top_chunk_ids": [result.chunk.id for result in results],
                "top_document_ids": [result.chunk.document_id for result in results],
                "top_sources": [
                    {
                        "chunk_id": result.chunk.id,
                        "document_id": result.chunk.document_id,
                        "title": result.chunk.title,
                        "score": round(result.score, 6),
                        "dense_score": round(result.dense_score, 6),
                        "keyword_score": round(result.keyword_score, 6),
                        "entity_score": round(result.entity_score, 6),
                        "document_score": round(result.document_score, 6),
                    }
                    for result in results
                ],
                **score,
            }
        )
    return {"summary": summarize_scores(rows), "by_category": summarize_scores_by_category(rows), "examples": rows}
