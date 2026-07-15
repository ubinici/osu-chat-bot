from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol
import math

from ..retrieval.analysis import ArtifactTopicResolver
from .datasets import QueryTopicExample, load_query_topic_dataset


class TextEmbedder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class TopicPredictor(Protocol):
    def predict(self, query: str, *, limit: int = 3) -> list["TopicPrediction"]: ...


@dataclass(frozen=True)
class TopicPrediction:
    topic_id: str
    score: float


class SemanticTopicKNN:
    """Small-data semantic baseline over reviewed training examples."""

    def __init__(self, examples: list[QueryTopicExample], embedder: TextEmbedder):
        if not examples:
            raise ValueError("SemanticTopicKNN needs at least one training example")
        self.examples = examples
        self.embedder = embedder
        vectors = embedder.encode([example.query for example in examples])
        if len(vectors) != len(examples):
            raise ValueError("Embedder returned a different number of vectors than queries")
        self.vectors = [_unit_vector(vector) for vector in vectors]

    def predict(self, query: str, *, limit: int = 3) -> list[TopicPrediction]:
        query_vectors = self.embedder.encode([query])
        if len(query_vectors) != 1:
            raise ValueError("Embedder must return exactly one vector for one query")
        query_vector = _unit_vector(query_vectors[0])
        scores: dict[str, float] = {}
        for example, vector in zip(self.examples, self.vectors, strict=True):
            similarity = _dot(query_vector, vector)
            for topic_id in example.topic_ids:
                scores[topic_id] = max(scores.get(topic_id, -1.0), similarity)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [
            TopicPrediction(topic_id=topic_id, score=score)
            for topic_id, score in ranked[: max(1, limit)]
        ]


class DeterministicTopicPredictor:
    def __init__(self, resolver: ArtifactTopicResolver):
        self.resolver = resolver

    def predict(self, query: str, *, limit: int = 3) -> list[TopicPrediction]:
        predictions: list[TopicPrediction] = []
        seen: set[str] = set()
        for topic in self.resolver.resolve(query):
            if topic.canonical_id in seen:
                continue
            seen.add(topic.canonical_id)
            predictions.append(TopicPrediction(topic.canonical_id, topic.confidence))
            if len(predictions) >= max(1, limit):
                break
        return predictions


def evaluate_topic_models(
    dataset_path: Path,
    alias_artifact: Path,
    embedder: TextEmbedder,
    *,
    split: str = "validation",
    top_k: int = 3,
    alias_minimum_confidence: float = 0.85,
    alias_minimum_tokens: int = 2,
) -> dict[str, object]:
    examples = load_query_topic_dataset(dataset_path)
    training_examples = [example for example in examples if example.split == "train"]
    evaluation_examples = [example for example in examples if example.split == split]
    if not evaluation_examples:
        raise ValueError(f"Dataset has no examples in split {split!r}")

    semantic = SemanticTopicKNN(training_examples, embedder)
    deterministic = DeterministicTopicPredictor(
        ArtifactTopicResolver(
            alias_artifact,
            minimum_confidence=alias_minimum_confidence,
            minimum_tokens=alias_minimum_tokens,
        )
    )
    deterministic_rows = _evaluate_predictor(evaluation_examples, deterministic, top_k=top_k)
    semantic_rows = _evaluate_predictor(evaluation_examples, semantic, top_k=top_k)
    return {
        "dataset": str(dataset_path),
        "alias_artifact": str(alias_artifact),
        "split": split,
        "training_examples": len(training_examples),
        "evaluation_examples": len(evaluation_examples),
        "top_k": top_k,
        "deterministic": _summarize(deterministic_rows),
        "semantic_knn": _summarize(semantic_rows),
        "examples": [
            {
                "query": example.query,
                "expected_topic_ids": example.topic_ids,
                "negative_topic_ids": example.negative_topic_ids,
                "deterministic": deterministic_row,
                "semantic_knn": semantic_row,
            }
            for example, deterministic_row, semantic_row in zip(
                evaluation_examples,
                deterministic_rows,
                semantic_rows,
                strict=True,
            )
        ],
    }


def _evaluate_predictor(
    examples: list[QueryTopicExample],
    predictor: TopicPredictor,
    *,
    top_k: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for example in examples:
        predictions = predictor.predict(example.query, limit=top_k)
        expected = set(example.topic_ids)
        matched_rank = next(
            (
                rank
                for rank, prediction in enumerate(predictions, start=1)
                if prediction.topic_id in expected
            ),
            0,
        )
        predicted_ids = [prediction.topic_id for prediction in predictions]
        rows.append(
            {
                "predictions": [asdict(prediction) for prediction in predictions],
                "matched_rank": matched_rank,
                "reciprocal_rank": 1.0 / matched_rank if matched_rank else 0.0,
                "hit_at_1": int(matched_rank == 1),
                "hit_at_3": int(0 < matched_rank <= 3),
                "covered": int(bool(predictions)),
                "hard_negative_at_1": int(
                    bool(predicted_ids) and predicted_ids[0] in set(example.negative_topic_ids)
                ),
            }
        )
    return rows


def _summarize(rows: list[dict[str, object]]) -> dict[str, float | int]:
    total = len(rows)
    return {
        "examples": total,
        "coverage": sum(int(row["covered"]) for row in rows) / total if total else 0.0,
        "hit_at_1": sum(int(row["hit_at_1"]) for row in rows) / total if total else 0.0,
        "hit_at_3": sum(int(row["hit_at_3"]) for row in rows) / total if total else 0.0,
        "mean_reciprocal_rank": (
            sum(float(row["reciprocal_rank"]) for row in rows) / total if total else 0.0
        ),
        "hard_negative_at_1": (
            sum(int(row["hard_negative_at_1"]) for row in rows) / total if total else 0.0
        ),
    }


def _unit_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(value) ** 2 for value in vector))
    if norm == 0:
        raise ValueError("Embedding vector must not be all zeroes")
    return [float(value) / norm for value in vector]


def _dot(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding vectors must have the same dimensions")
    return sum(a * b for a, b in zip(left, right, strict=True))
