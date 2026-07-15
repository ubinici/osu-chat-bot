import json

import pytest

from osu_chatbot.learning.datasets import (
    load_query_topic_dataset,
    promote_feedback_dataset,
)


def write_rows(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_promotes_only_reviewed_accepted_feedback_and_strips_runtime_context(tmp_path) -> None:
    feedback = tmp_path / "feedback.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    output = tmp_path / "query_topics.jsonl"
    write_rows(
        feedback,
        [
            {
                "event_id": "discord:1",
                "occurred_at": "2026-07-15T12:00:00Z",
                "query": "Why are my hit windows so strict?",
                "feedback": "correction",
                "source": "discord",
                "analysis": {"topics": []},
                "retrieved_chunk_ids": ["Beatmapping::article"],
                "retrieved_document_ids": ["Beatmapping"],
                "answer_version": "rag-v2",
            },
            {
                "event_id": "discord:2",
                "occurred_at": "2026-07-15T12:01:00Z",
                "query": "Unreviewed query",
                "feedback": "negative",
                "source": "discord",
            },
            {
                "event_id": "discord:3",
                "occurred_at": "2026-07-15T12:02:00Z",
                "query": "Bad annotation",
                "feedback": "negative",
                "source": "discord",
            },
        ],
    )
    write_rows(
        reviews,
        [
            {
                "event_id": "discord:1",
                "decision": "accept",
                "topic_ids": ["Beatmap/Overall_difficulty"],
                "negative_topic_ids": ["Beatmapping/Timing"],
                "split": "train",
            },
            {"event_id": "discord:3", "decision": "reject"},
        ],
    )

    report = promote_feedback_dataset(feedback, reviews, output)

    assert report == {
        "feedback_events": 3,
        "reviews": 2,
        "promoted_examples": 1,
        "rejected_reviews": 1,
        "excluded_held_out": 0,
        "unreviewed_events": 1,
    }
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "query": "Why are my hit windows so strict?",
            "topic_ids": ["Beatmap/Overall_difficulty"],
            "split": "train",
            "source": "reviewed_discord_feedback",
            "source_event_id": "discord:1",
            "negative_topic_ids": ["Beatmapping/Timing"],
        }
    ]
    assert "analysis" not in rows[0]
    assert "retrieved_chunk_ids" not in rows[0]


def test_excludes_normalized_held_out_evaluation_queries(tmp_path) -> None:
    feedback = tmp_path / "feedback.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    held_out = tmp_path / "eval.jsonl"
    output = tmp_path / "query_topics.jsonl"
    write_rows(
        feedback,
        [
            {
                "event_id": "discord:1",
                "occurred_at": "2026-07-15T12:00:00Z",
                "query": "What is PP?",
                "feedback": "positive",
                "source": "discord",
            }
        ],
    )
    write_rows(
        reviews,
        [{"event_id": "discord:1", "decision": "accept", "topic_ids": ["Performance_points"]}],
    )
    write_rows(held_out, [{"question": "what is pp"}])

    report = promote_feedback_dataset(feedback, reviews, output, held_out_path=held_out)

    assert report["promoted_examples"] == 0
    assert report["excluded_held_out"] == 1
    assert output.read_text(encoding="utf-8") == ""


def test_rejects_duplicate_query_across_training_splits(tmp_path) -> None:
    dataset = tmp_path / "query_topics.jsonl"
    write_rows(
        dataset,
        [
            {
                "query": "What is AR?",
                "topic_ids": ["Beatmap/Approach_rate"],
                "split": "train",
                "source": "seed",
                "source_event_id": "seed:1",
            },
            {
                "query": "what is ar",
                "topic_ids": ["Beatmap/Approach_rate"],
                "split": "validation",
                "source": "seed",
                "source_event_id": "seed:2",
            },
        ],
    )

    with pytest.raises(ValueError, match="Conflicting labels or splits"):
        load_query_topic_dataset(dataset)
