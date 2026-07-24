from __future__ import annotations

from ..domain.models import SearchResult


def build_prompt(
    question: str,
    results: list[SearchResult],
    *,
    style: str | None = None,
    history: list[tuple[str, str]] | None = None,
) -> str:
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
    conversation = ""
    if history:
        turns = "\n".join(
            f"User: {user}\nAssistant: {assistant}"
            for user, assistant in history
        )
        conversation = (
            "Conversation history (use only to resolve references and maintain continuity; "
            "it is not factual evidence):\n"
            f"{turns}\n\n"
        )
    return (
        "You are a friendly osu! assistant. Answer in a clear, natural, conversational tone.\n"
        f"{style or 'Match the user formality without forcing slang.'}\n"
        "Use only the cited osu! wiki/news context below.\n"
        "If the context does not support the answer, say that the provided osu! wiki/news context is insufficient.\n"
        "Never rely on or repeat citations from conversation history; cite only the current context.\n"
        "Cite every factual claim with bracket citations like [1] or [2]. Do not cite sources that do not support the claim.\n\n"
        f"{conversation}"
        f"Current question: {question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )
