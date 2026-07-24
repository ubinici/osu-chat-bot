from __future__ import annotations

from ..domain.models import SearchResult
from .base import TextGenerator
from .prompt import build_prompt

INSUFFICIENT_CONTEXT = "I do not have enough supported osu! wiki/news context to answer that accurately."


def answer_question(
    question: str,
    results: list[SearchResult],
    generator: TextGenerator,
    *,
    style: str | None = None,
    history: list[tuple[str, str]] | None = None,
) -> str:
    if not results:
        return INSUFFICIENT_CONTEXT
    answer = generator.generate(
        build_prompt(question, results, style=style, history=history)
    )
    return answer or INSUFFICIENT_CONTEXT
