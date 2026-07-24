from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from threading import Lock

from .datasets import FeedbackEvent


class FeedbackEventStore:
    """Thread-safe append store for one Discord bot process."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = Lock()

    def append(self, event: FeedbackEvent) -> None:
        payload = json.dumps(asdict(event), ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(payload + "\n")
                handle.flush()
