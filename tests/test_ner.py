from pathlib import Path

from osu_chatbot.domain.artifacts import ENTITY_CANDIDATES_FILE, write_jsonl
from osu_chatbot.domain.models import Chunk
from osu_chatbot.knowledge.label_profiles import labels_for_chunk, labels_for_profile
from osu_chatbot.knowledge.ner import build_entity_candidates_from_artifacts, extract_entity_candidates, select_chunks


class FakeExtractor:
    backend = "fake"
    source_model = "fake-model"

    def extract(self, text: str, labels: list[str]) -> list[dict]:
        start = text.index("Hidden")
        return [
            {
                "text": "Hidden",
                "label": labels[0],
                "score": 0.91,
                "start": start,
                "end": start + len("Hidden"),
            },
            {
                "text": "game",
                "label": labels[0],
                "score": 0.99,
                "start": 0,
                "end": 4,
            },
        ]


def test_extract_entity_candidates_filters_and_preserves_context() -> None:
    chunks = [
        Chunk(
            id="mods::section",
            document_id="Gameplay/Game_modifier/Hidden",
            source_type="wiki",
            file_path="Gameplay/Game_modifier/Hidden/en.md",
            osu_url="https://osu.ppy.sh/wiki/en/Gameplay/Game_modifier/Hidden",
            title="Hidden",
            text="The Hidden game modifier removes approach circles.",
            chunk_index=0,
            heading_path=["Gameplay", "Game modifier", "Hidden"],
        )
    ]

    candidates = extract_entity_candidates(chunks, FakeExtractor(), labels=["game modifier"])

    assert len(candidates) == 1
    assert candidates[0].text == "Hidden"
    assert candidates[0].label == "game modifier"
    assert "approach circles" in candidates[0].context


def test_main_page_profile_scopes_wiki_style_chunks_to_wiki_maintenance() -> None:
    chunk = Chunk(
        id="style::section",
        document_id="Article_styling_criteria/Formatting",
        source_type="wiki",
        file_path="Article_styling_criteria/Formatting/en.md",
        osu_url="https://osu.ppy.sh/wiki/en/Article_styling_criteria/Formatting",
        title="Formatting",
        text="Line breaks must use a backslash.",
        chunk_index=0,
    )

    labels = labels_for_chunk(chunk, labels_for_profile("main-page"), profile="main-page", scoped=True)

    assert labels[0] == "osu wiki maintenance or contribution term"
    assert "osu beatmap editor tool or mapping term" in labels


def test_generic_wiki_jargon_is_filtered() -> None:
    chunks = [
        Chunk(
            id="style::section",
            document_id="Article_styling_criteria/Formatting",
            source_type="wiki",
            file_path="Article_styling_criteria/Formatting/en.md",
            osu_url="https://osu.ppy.sh/wiki/en/Article_styling_criteria/Formatting",
            title="Formatting",
            text="Line breaks are wiki formatting. Beatmap discussions is an osu! tag.",
            chunk_index=0,
        )
    ]

    class WikiJargonExtractor:
        backend = "fake"
        source_model = "fake-model"

        def extract(self, text: str, labels: list[str]) -> list[dict]:
            return [
                {"text": "topic", "label": labels[0], "score": 0.9, "start": 0, "end": 5},
                {"text": "disambiguation", "label": labels[0], "score": 0.9, "start": 0, "end": 14},
                {"text": "beatmap discussions", "label": "osu beatmap editor tool or mapping term", "score": 0.9, "start": 37, "end": 56},
            ]

    candidates = extract_entity_candidates(
        chunks,
        WikiJargonExtractor(),
        labels=labels_for_profile("main-page"),
        scoped_labels=True,
    )

    assert [candidate.text for candidate in candidates] == ["beatmap discussions"]


def test_balanced_chunk_selection_spreads_limit_across_documents() -> None:
    chunks = [
        Chunk(id="a1", document_id="A", source_type="wiki", file_path="A/en.md", osu_url="", title="A", text="A1", chunk_index=0),
        Chunk(id="a2", document_id="A", source_type="wiki", file_path="A/en.md", osu_url="", title="A", text="A2", chunk_index=1),
        Chunk(id="a3", document_id="A", source_type="wiki", file_path="A/en.md", osu_url="", title="A", text="A3", chunk_index=2),
        Chunk(id="b1", document_id="B", source_type="wiki", file_path="B/en.md", osu_url="", title="B", text="B1", chunk_index=0),
        Chunk(id="c1", document_id="C", source_type="wiki", file_path="C/en.md", osu_url="", title="C", text="C1", chunk_index=0),
    ]

    selected = select_chunks(chunks, limit=3, sampling="balanced")

    assert [chunk.id for chunk in selected] == ["a1", "b1", "c1"]


def test_build_entity_candidates_from_artifacts_writes_candidate_artifact(tmp_path: Path, monkeypatch) -> None:
    artifact_dir = tmp_path / "rag"
    output_dir = tmp_path / "runs" / "ner"
    write_jsonl(
        artifact_dir / "chunks_hierarchical.jsonl",
        [
            Chunk(
                id="mods::section",
                document_id="Gameplay/Game_modifier/Hidden",
                source_type="wiki",
                file_path="Gameplay/Game_modifier/Hidden/en.md",
                osu_url="https://osu.ppy.sh/wiki/en/Gameplay/Game_modifier/Hidden",
                title="Hidden",
                text="The Hidden game modifier removes approach circles.",
                chunk_index=0,
            )
        ],
    )
    monkeypatch.setattr("osu_chatbot.knowledge.ner.create_extractor", lambda *args, **kwargs: FakeExtractor())

    report = build_entity_candidates_from_artifacts(
        artifact_dir,
        output_dir=output_dir,
        backend="gliner",
        labels=["game modifier"],
        label_profile="main-page",
        threshold=0.5,
    )

    assert report["candidates"] == 1
    assert not (artifact_dir / ENTITY_CANDIDATES_FILE).exists()
    assert (output_dir / ENTITY_CANDIDATES_FILE).exists()
