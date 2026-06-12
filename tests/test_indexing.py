from pathlib import Path

from osu_chatbot.domain.artifacts import write_json
from osu_chatbot.domain.models import Chunk
from osu_chatbot.indexing.pipeline import IndexOptions, embedding_text_for_chunk, file_fingerprint, resolve_start_offset


def test_resume_uses_saved_offset_when_chunk_artifact_matches(tmp_path: Path) -> None:
    chunk_artifact = tmp_path / "chunks_hierarchical.jsonl"
    chunk_artifact.write_text('{"id": "one"}\n', encoding="utf-8")
    fingerprint = file_fingerprint(chunk_artifact)
    write_json(
        tmp_path / "index_state.json",
        {
            "next_offset": 128,
            "artifact_fingerprint": fingerprint,
        },
    )

    assert resolve_start_offset(tmp_path, IndexOptions(offset=0, resume=True), fingerprint) == 128


def test_resume_falls_back_to_requested_offset_when_artifact_changed(tmp_path: Path) -> None:
    write_json(
        tmp_path / "index_state.json",
        {
            "next_offset": 128,
            "artifact_fingerprint": {"size": 1, "mtime_ns": 1},
        },
    )

    assert resolve_start_offset(tmp_path, IndexOptions(offset=16, resume=True), {"size": 2, "mtime_ns": 2}) == 16


def test_embedding_text_for_chunk_includes_retrieval_metadata() -> None:
    chunk = Chunk(
        id="beatmap-approach-rate",
        document_id="Beatmap/Approach_rate",
        source_type="wiki",
        file_path="Beatmap/Approach_rate/en.md",
        osu_url="https://osu.ppy.sh/wiki/en/Beatmap/Approach_rate",
        title="Approach rate",
        text="Approach rate controls how long hit objects are visible before they need to be hit.",
        chunk_index=0,
        heading_path=["Approach rate", "Mod effects"],
        tags=["AR", "difficulty"],
        metadata={"chunk_type": "section", "domain": "beatmap", "subculture": "mapping"},
    )

    text = embedding_text_for_chunk(chunk)

    assert "Title: Approach rate" in text
    assert "Headings: Approach rate > Mod effects" in text
    assert "Tags: AR, difficulty" in text
    assert "Domain: beatmap" in text
    assert chunk.text in text
