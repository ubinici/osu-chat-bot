from __future__ import annotations

from typing import Protocol


class GenerationError(RuntimeError):
    """A configured text-generation provider could not return an answer."""


class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str: ...
