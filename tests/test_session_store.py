import sqlite3
from concurrent.futures import ThreadPoolExecutor

from osu_chatbot.learning.session_store import DiscordSessionStore


def test_session_store_persists_and_finalizes_complete_transcript(tmp_path) -> None:
    path = tmp_path / "discord_sessions.sqlite3"
    store = DiscordSessionStore(path)
    store.start_session(
        "session-1",
        answer_version="gpt-oss-discord-v1",
        created_at="2026-07-24T10:00:00+00:00",
    )
    store.append_turn(
        "session-1",
        turn_index=1,
        user_message="What does AR do?",
        assistant_message="It controls object visibility time. [1]",
        response={
            "answer": "It controls object visibility time. [1]",
            "sources": [{"document_id": "Beatmap/Approach_rate"}],
            "latency_ms": 42,
        },
        occurred_at="2026-07-24T10:00:10+00:00",
    )

    assert store.close_session(
        "session-1",
        reason="inactivity",
        closed_at="2026-07-24T10:05:10+00:00",
    )

    session = store.load_session("session-1")
    assert session is not None
    assert session.close_reason == "inactivity"
    assert session.closed_at == "2026-07-24T10:05:10+00:00"
    assert session.turns[0].user_message == "What does AR do?"
    assert session.turns[0].assistant_message.endswith("[1]")
    assert session.turns[0].response["latency_ms"] == 42
    assert "answer" not in session.turns[0].response

    with sqlite3.connect(path) as connection:
        session_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(chat_sessions)")
        }
        turn_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(chat_turns)")
        }
    assert "user_id" not in session_columns
    assert "owner_id" not in session_columns
    assert "user_id" not in turn_columns


def test_session_store_closes_interrupted_open_sessions(tmp_path) -> None:
    store = DiscordSessionStore(tmp_path / "discord_sessions.sqlite3")
    store.start_session("open-1", answer_version="v1")
    store.start_session("open-2", answer_version="v1")
    store.start_session("closed", answer_version="v1")
    store.close_session("closed", reason="user_closed")

    interrupted = store.close_all_open_sessions(reason="bot_restart")

    assert set(interrupted) == {"open-1", "open-2"}
    assert store.load_session("open-1").close_reason == "bot_restart"
    assert store.load_session("closed").close_reason == "user_closed"


def test_session_store_refuses_to_finalize_unknown_session(tmp_path) -> None:
    store = DiscordSessionStore(tmp_path / "discord_sessions.sqlite3")

    try:
        store.close_session("missing", reason="inactivity")
    except ValueError as exc:
        assert "Unknown chat session" in str(exc)
    else:
        raise AssertionError("a missing transcript must not be treated as safely finalized")


def test_session_store_accepts_turns_from_four_rooms_concurrently(tmp_path) -> None:
    store = DiscordSessionStore(tmp_path / "discord_sessions.sqlite3")
    for room in range(4):
        store.start_session(f"room-{room}", answer_version="v1")

    def append_room_turns(room: int) -> None:
        for turn in range(1, 6):
            store.append_turn(
                f"room-{room}",
                turn_index=turn,
                user_message=f"question {turn}",
                assistant_message=f"answer {turn}",
                response={"sources": []},
            )

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(append_room_turns, range(4)))

    assert all(len(store.load_session(f"room-{room}").turns) == 5 for room in range(4))
