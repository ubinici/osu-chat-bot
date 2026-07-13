from pathlib import Path

from osu_chatbot.domain.artifacts import DOCUMENT_ALIASES_FILE, read_jsonl, write_jsonl
from osu_chatbot.domain.models import Chunk
from osu_chatbot.knowledge.aliases import build_document_alias_artifacts, build_document_alias_rows


def document(page_id: str, title: str, *, source: str = "osu-wiki") -> dict:
    return {
        "source": source,
        "page_id": page_id,
        "title": title,
        "repo_rel_path": f"{page_id}/en.md",
        "tags": [],
        "sections": [{"title": title, "heading_path": [title]}],
    }


def test_document_alias_rows_include_title_and_chunk_acronyms() -> None:
    docs = [document("Beatmap/Approach_rate", "Approach rate")]
    chunks = [
        Chunk(
            id="ar::article",
            document_id="Beatmap/Approach_rate",
            source_type="wiki",
            file_path="Beatmap/Approach_rate/en.md",
            osu_url="https://osu.ppy.sh/wiki/en/Beatmap/Approach_rate",
            title="Approach rate",
            text="**Approach rate** (***AR***) controls how soon objects appear.",
            chunk_index=0,
            heading_path=["Approach rate"],
        )
    ]

    rows = build_document_alias_rows(docs, chunks, [])
    alias_keys = {(row["alias_key"], row["source"]) for row in rows}

    assert ("approach rate", "title") in alias_keys
    assert ("ar", "title_acronym") in alias_keys or ("ar", "chunk_acronym") in alias_keys
    assert {row["topic_id"] for row in rows} == {"Beatmap/Approach_rate"}
    assert rows[0]["retrieval_lane"] == "canonical"
    assert rows[0]["trust_tier"] == "canonical"


def test_document_alias_artifact_uses_only_accepted_link_aliases(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "rag"
    docs = [document("Target/Page", "Target page")]
    chunks = [
        Chunk(
            id="target::article",
            document_id="Target/Page",
            source_type="wiki",
            file_path="Target/Page/en.md",
            osu_url="https://osu.ppy.sh/wiki/en/Target/Page",
            title="Target page",
            text="Target page.",
            chunk_index=0,
            heading_path=["Target page"],
        )
    ]
    write_jsonl(artifact_dir / "documents_structured.jsonl", docs)
    write_jsonl(artifact_dir / "chunks_hierarchical.jsonl", chunks)
    write_jsonl(
        artifact_dir / "link_alias_candidates.jsonl",
        [
            {
                "alias": "accepted alias",
                "target_page_id": "Target/Page",
                "decision": "accept",
                "confidence": 0.95,
            },
            {
                "alias": "review alias",
                "target_page_id": "Target/Page",
                "decision": "review",
                "confidence": 0.95,
            },
        ],
    )

    report = build_document_alias_artifacts(artifact_dir)
    rows = read_jsonl(artifact_dir / DOCUMENT_ALIASES_FILE)
    accepted = [row for row in rows if row["alias_key"] == "accepted alias"]
    reviewed = [row for row in rows if row["alias_key"] == "review alias"]

    assert report["aliases"] == len(rows)
    assert report["topics"] == 1
    assert accepted[0]["source"] == "accepted_link"
    assert reviewed == []


def test_alias_rows_include_topic_identifier_aliases() -> None:
    docs = [document("Beatmap/HP_drain_rate", "HP drain rate")]

    rows = build_document_alias_rows(docs, [], [])
    health_rows = [row for row in rows if row["alias_key"] == "beatmap health drain"]
    identifier_rows = [row for row in health_rows if row["source"] == "topic_identifier"]

    assert health_rows
    assert identifier_rows
    assert identifier_rows[0]["document_id"] == "Beatmap/HP_drain_rate"
    assert identifier_rows[0]["topic_id"] == "Beatmap/Health_drain"
    assert "Beatmap/Health_drain" in identifier_rows[0]["topic_document_ids"]
