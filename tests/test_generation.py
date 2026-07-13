from osu_chatbot.domain.models import Chunk, SearchResult
from osu_chatbot.generation.prompt import build_prompt


def test_prompt_requests_grounded_conversational_answer_with_citations() -> None:
    result = SearchResult(
        chunk=Chunk(
            id="ar::article",
            document_id="Beatmap/Approach_rate",
            source_type="wiki",
            file_path="Beatmap/Approach_rate/en.md",
            osu_url="https://osu.ppy.sh/wiki/en/Beatmap/Approach_rate",
            title="Approach rate",
            text="Approach rate controls how long hit objects are visible.",
            chunk_index=0,
            heading_path=["Approach rate"],
        ),
        score=0.9,
    )

    prompt = build_prompt("what does AR do?", [result])

    assert "natural, conversational tone" in prompt
    assert "using only" not in prompt
    assert "Use only the cited" in prompt
    assert "[1]" in prompt
    assert result.chunk.osu_url in prompt
