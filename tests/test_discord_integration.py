import json

from osu_chatbot.integrations.discord_bot import (
    format_discord_response,
    make_feedback_event,
    split_discord_text,
)
from osu_chatbot.learning.datasets import load_feedback_events
from osu_chatbot.learning.feedback_store import FeedbackEventStore


def _payload() -> dict:
    return {
        "answer": "AR changes how long objects are visible. [1]",
        "sources": [
            {
                "citation": 1,
                "chunk_id": "Beatmap/Approach_rate::article",
                "document_id": "Beatmap/Approach_rate",
                "title": "Approach rate",
                "url": "https://osu.ppy.sh/wiki/en/Beatmap/Approach_rate",
            }
        ],
        "intent": ["definition"],
        "search_query": "what does AR do?",
        "latency_ms": 42,
        "retrieval_lane": "canonical",
        "resolved_topics": [],
        "response_type": "answer",
    }


def test_discord_response_includes_clickable_sources_and_respects_limit() -> None:
    payload = _payload()
    payload["answer"] = ("A useful sentence. " * 180).strip() + " [1]"

    messages = format_discord_response(payload)

    assert len(messages) > 1
    assert all(len(message) <= 1900 for message in messages)
    assert "Approach rate" in messages[-1]
    assert "https://osu.ppy.sh/wiki/en/Beatmap/Approach_rate" in messages[-1]


def test_discord_text_split_handles_single_oversized_word() -> None:
    chunks = split_discord_text("x" * 55, limit=20)

    assert chunks == ["x" * 20, "x" * 20, "x" * 15]


def test_feedback_event_store_writes_promotion_compatible_record_without_user_id(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    event = make_feedback_event(
        event_id="discord:123",
        question="what does AR do?",
        payload=_payload(),
        feedback="negative",
        reason="wrong_answer",
        comment="It should mention visibility time.",
        answer_version="gpt-oss-discord-v1",
    )

    FeedbackEventStore(path).append(event)

    loaded = load_feedback_events(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == [event]
    assert loaded[0].retrieved_document_ids == ["Beatmap/Approach_rate"]
    assert loaded[0].analysis["feedback_reason"] == "wrong_answer"
    assert "user_id" not in raw

