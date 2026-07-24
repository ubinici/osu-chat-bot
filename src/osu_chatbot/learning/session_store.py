from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any


@dataclass(frozen=True)
class StoredTurn:
    turn_index: int
    occurred_at: str
    user_message: str
    assistant_message: str
    response: dict[str, Any]


@dataclass(frozen=True)
class StoredSession:
    session_id: str
    created_at: str
    closed_at: str | None
    close_reason: str | None
    answer_version: str
    turns: list[StoredTurn]


class DiscordSessionStore:
    """SQLite transcript store with no durable Discord user identifier."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def start_session(
        self,
        session_id: str,
        *,
        answer_version: str,
        created_at: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO chat_sessions (
                    session_id, created_at, answer_version, transcript_policy
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    session_id,
                    created_at or _utc_now(),
                    answer_version,
                    "private_room_evaluation_v1",
                ),
            )

    def append_turn(
        self,
        session_id: str,
        *,
        turn_index: int,
        user_message: str,
        assistant_message: str,
        response: dict[str, Any],
        occurred_at: str | None = None,
    ) -> None:
        if turn_index < 1:
            raise ValueError("turn_index must be positive")
        metadata = dict(response)
        metadata.pop("answer", None)
        with self._connect() as connection:
            session = connection.execute(
                "SELECT closed_at FROM chat_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                raise ValueError(f"Unknown chat session {session_id!r}")
            if session[0] is not None:
                raise ValueError(f"Chat session {session_id!r} is already closed")
            connection.execute(
                """
                INSERT INTO chat_turns (
                    session_id, turn_index, occurred_at, user_message,
                    assistant_message, response_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    turn_index,
                    occurred_at or _utc_now(),
                    user_message,
                    assistant_message,
                    json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
                ),
            )

    def close_session(
        self,
        session_id: str,
        *,
        reason: str,
        closed_at: str | None = None,
    ) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE chat_sessions
                SET closed_at = ?, close_reason = ?
                WHERE session_id = ? AND closed_at IS NULL
                """,
                (closed_at or _utc_now(), reason, session_id),
            )
            if cursor.rowcount == 0:
                exists = connection.execute(
                    "SELECT 1 FROM chat_sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if exists is None:
                    raise ValueError(f"Unknown chat session {session_id!r}")
        return cursor.rowcount == 1

    def close_all_open_sessions(self, *, reason: str) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT session_id FROM chat_sessions WHERE closed_at IS NULL"
            ).fetchall()
            session_ids = [str(row[0]) for row in rows]
            if session_ids:
                connection.execute(
                    """
                    UPDATE chat_sessions
                    SET closed_at = ?, close_reason = ?
                    WHERE closed_at IS NULL
                    """,
                    (_utc_now(), reason),
                )
        return session_ids

    def load_session(self, session_id: str) -> StoredSession | None:
        with self._connect() as connection:
            session = connection.execute(
                """
                SELECT session_id, created_at, closed_at, close_reason, answer_version
                FROM chat_sessions WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            if session is None:
                return None
            rows = connection.execute(
                """
                SELECT turn_index, occurred_at, user_message,
                       assistant_message, response_json
                FROM chat_turns
                WHERE session_id = ?
                ORDER BY turn_index
                """,
                (session_id,),
            ).fetchall()
        return StoredSession(
            session_id=str(session[0]),
            created_at=str(session[1]),
            closed_at=str(session[2]) if session[2] is not None else None,
            close_reason=str(session[3]) if session[3] is not None else None,
            answer_version=str(session[4]),
            turns=[
                StoredTurn(
                    turn_index=int(row[0]),
                    occurred_at=str(row[1]),
                    user_message=str(row[2]),
                    assistant_message=str(row[3]),
                    response=json.loads(row[4]),
                )
                for row in rows
            ],
        )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    closed_at TEXT,
                    close_reason TEXT,
                    answer_version TEXT NOT NULL,
                    transcript_policy TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chat_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    occurred_at TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    assistant_message TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    UNIQUE(session_id, turn_index),
                    FOREIGN KEY(session_id) REFERENCES chat_sessions(session_id)
                );

                CREATE INDEX IF NOT EXISTS idx_chat_turns_session
                ON chat_turns(session_id, turn_index);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
