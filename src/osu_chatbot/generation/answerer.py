from __future__ import annotations

from ..config import OllamaConfig
from ..domain.models import SearchResult
from .ollama import generate_with_ollama
from .prompt import build_prompt

INSUFFICIENT_CONTEXT = "I do not have enough supported osu! wiki/news context to answer that accurately."


def answer_question(question: str, results: list[SearchResult], config: OllamaConfig) -> str:
    if not results:
        return INSUFFICIENT_CONTEXT
    answer = generate_with_ollama(build_prompt(question, results), config)
    return answer or INSUFFICIENT_CONTEXT
