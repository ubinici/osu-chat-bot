import json

from osu_chatbot.integrations.discord_bot import (
    ActiveChatSession,
    ChatApiClient,
    ChatRoomRegistry,
    format_discord_response,
    managed_session_id,
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


def test_discord_api_client_sends_only_supplied_room_history(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(_payload()).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(
        "osu_chatbot.integrations.discord_bot.urlopen",
        fake_urlopen,
    )

    response = ChatApiClient("http://app:8000", timeout_seconds=30).ask(
        "Does it affect every mode?",
        history=[("What does OD do?", "It changes hit windows. [1]")],
    )

    assert response["answer"].endswith("[1]")
    assert captured == {
        "url": "http://app:8000/v1/chat",
        "payload": {
            "question": "Does it affect every mode?",
            "history": [
                {
                    "user": "What does OD do?",
                    "assistant": "It changes hit windows. [1]",
                }
            ],
        },
        "timeout": 30,
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
        session_id="session-1",
    )

    FeedbackEventStore(path).append(event)

    loaded = load_feedback_events(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == [event]
    assert loaded[0].retrieved_document_ids == ["Beatmap/Approach_rate"]
    assert loaded[0].analysis["feedback_reason"] == "wrong_answer"
    assert loaded[0].analysis["session_id"] == "session-1"
    assert "user_id" not in raw


def test_room_registry_enforces_capacity_and_one_room_per_owner() -> None:
    rooms = ChatRoomRegistry(max_active_rooms=2)
    first = ActiveChatSession(session_id="one", owner_id=10, channel_id=100)
    second = ActiveChatSession(session_id="two", owner_id=20, channel_id=200)

    rooms.add(first)
    rooms.add(second)

    assert rooms.at_capacity()
    assert rooms.get(100) is first
    assert rooms.channel_for_owner(20) == 200

    try:
        rooms.add(ActiveChatSession(session_id="three", owner_id=30, channel_id=300))
    except ValueError as exc:
        assert "currently in use" in str(exc)
    else:
        raise AssertionError("capacity must be enforced")

    rooms.remove(100)
    try:
        rooms.add(ActiveChatSession(session_id="duplicate", owner_id=20, channel_id=300))
    except ValueError as exc:
        assert "already owns" in str(exc)
    else:
        raise AssertionError("one active room per owner must be enforced")


def test_active_session_context_is_room_local_and_bounded() -> None:
    first = ActiveChatSession(session_id="one", owner_id=10, channel_id=100)
    second = ActiveChatSession(session_id="two", owner_id=20, channel_id=200)
    first.turns.extend([("q1", "a1"), ("q2", "a2"), ("q3", "a3")])
    second.turns.append(("private", "other room"))

    assert first.context(2) == [("q2", "a2"), ("q3", "a3")]
    assert second.context(2) == [("private", "other room")]
    assert managed_session_id("osu-chat-session:abc123") == "abc123"
    assert managed_session_id("an ordinary channel") is None


def test_active_session_becomes_inactive_at_configured_threshold() -> None:
    session = ActiveChatSession(
        session_id="one",
        owner_id=10,
        channel_id=100,
        last_activity=100.0,
    )

    assert not session.is_inactive(now=399.9, timeout_seconds=300)
    assert session.is_inactive(now=400.0, timeout_seconds=300)
    session.busy = True
    assert not session.is_inactive(now=1000.0, timeout_seconds=300)
