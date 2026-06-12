from __future__ import annotations

from ..domain.models import SearchResult


def build_prompt(question: str, results: list[SearchResult]) -> str:
    context_parts = []
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        heading = " > ".join(chunk.heading_path) if chunk.heading_path else chunk.title
        context_parts.append(
            f"[{index}] {chunk.title} | {heading}\n"
            f"URL: {chunk.osu_url}\n"
            f"Text:\n{chunk.text}"
        )

    context = "\n\n---\n\n".join(context_parts)
    return (
        "You answer osu!-related questions using only the cited osu! wiki/news context below.\n"
        "If the context does not support the answer, say that the provided osu! wiki/news context is insufficient.\n"
        "Cite every factual claim with bracket citations like [1] or [2]. Do not cite sources that do not support the claim.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )
