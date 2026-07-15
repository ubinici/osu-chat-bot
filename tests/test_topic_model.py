import json

from osu_chatbot.learning.datasets import QueryTopicExample
from osu_chatbot.learning.topic_model import SemanticTopicKNN, evaluate_topic_models


class KeywordEmbedder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            normalized = text.casefold()
            vectors.append(
                [
                    float("strict" in normalized or "judgement" in normalized),
                    float("warning" in normalized or "circles" in normalized),
                    0.1,
                ]
            )
        return vectors


def write_rows(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_semantic_topic_knn_ranks_topics_from_reviewed_examples() -> None:
    model = SemanticTopicKNN(
        [
            QueryTopicExample(
                query="strict hit windows",
                topic_ids=["Beatmap/Overall_difficulty"],
                split="train",
                source="seed",
                source_event_id="seed:1",
            ),
            QueryTopicExample(
                query="circles appear with little warning",
                topic_ids=["Beatmap/Approach_rate"],
                split="train",
                source="seed",
                source_event_id="seed:2",
            ),
        ],
        KeywordEmbedder(),
    )

    predictions = model.predict("judgement timing is strict", limit=2)

    assert [prediction.topic_id for prediction in predictions] == [
        "Beatmap/Overall_difficulty",
        "Beatmap/Approach_rate",
    ]


def test_topic_model_report_compares_semantic_and_deterministic_predictions(tmp_path) -> None:
    dataset = tmp_path / "topics.jsonl"
    aliases = tmp_path / "aliases.jsonl"
    write_rows(
        dataset,
        [
            {
                "query": "strict hit windows",
                "topic_ids": ["Beatmap/Overall_difficulty"],
                "split": "train",
                "source": "seed",
                "source_event_id": "seed:1",
            },
            {
                "query": "circles appear with little warning",
                "topic_ids": ["Beatmap/Approach_rate"],
                "split": "train",
                "source": "seed",
                "source_event_id": "seed:2",
            },
            {
                "query": "judgement timing feels strict",
                "topic_ids": ["Beatmap/Overall_difficulty"],
                "negative_topic_ids": ["Beatmapping/Timing"],
                "split": "validation",
                "source": "seed",
                "source_event_id": "seed:3",
            },
        ],
    )
    write_rows(
        aliases,
        [
            {
                "alias": "overall difficulty",
                "document_id": "Beatmap/Overall_difficulty",
                "confidence": 1.0,
            }
        ],
    )

    report = evaluate_topic_models(dataset, aliases, KeywordEmbedder())

    assert report["deterministic"]["coverage"] == 0.0
    assert report["deterministic"]["hit_at_1"] == 0.0
    assert report["semantic_knn"]["coverage"] == 1.0
    assert report["semantic_knn"]["hit_at_1"] == 1.0
    assert report["semantic_knn"]["hard_negative_at_1"] == 0.0
