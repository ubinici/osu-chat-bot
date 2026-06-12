from pathlib import Path

from osu_chatbot.domain.artifacts import (
    DOCUMENTS_FILE,
    ENTITY_CANDIDATES_FILE,
    ENTITY_NORMALIZATION_REVIEW_FILE,
    write_jsonl,
)
from osu_chatbot.knowledge.normalization import (
    build_document_index,
    build_entity_normalization_artifacts,
    normalize_entity_candidates,
)


def test_normalize_candidates_links_plural_mentions_to_source_article() -> None:
    documents = [
        {
            "source": "osu-wiki",
            "page_id": "Beatmap/Pattern/osu!/Jump",
            "title": "Jump",
            "repo_rel_path": "Beatmap/Pattern/osu!/Jump/en.md",
        }
    ]
    candidates = [
        candidate("Jumps", "Beatmap/Pattern/osu!/Jump", score=0.93),
    ]

    rows = normalize_entity_candidates(candidates, build_document_index(documents))

    assert rows[0]["decision"] == "accept"
    assert rows[0]["canonical"] == "Jump"
    assert rows[0]["target_page_id"] == "Beatmap/Pattern/osu!/Jump"


def test_normalize_candidates_uses_manual_alias_targets() -> None:
    documents = [
        {
            "source": "osu-wiki",
            "page_id": "Beatmap/Approach_rate",
            "title": "Approach rate",
            "repo_rel_path": "Beatmap/Approach_rate/en.md",
        },
        {
            "source": "osu-wiki",
            "page_id": "Beatmapping/Beatmap_submission",
            "title": "Submission",
            "repo_rel_path": "Beatmapping/Beatmap_submission/en.md",
        },
    ]
    candidates = [
        candidate("AR", "Beatmap/Approach_rate", score=0.52),
        candidate("BSS", "Beatmapping/Beatmap_submission", score=0.55),
    ]

    rows = normalize_entity_candidates(candidates, build_document_index(documents))
    by_alias = {row["example_text"]: row for row in rows}

    assert by_alias["AR"]["canonical"] == "Approach rate"
    assert by_alias["AR"]["decision"] == "review"
    assert by_alias["BSS"]["target_page_id"] == "Beatmapping/Beatmap_submission"


def test_normalize_candidates_rejects_generic_domain_words() -> None:
    rows = normalize_entity_candidates(
        [candidate("gameplay", "Beatmap/Background", score=0.9)],
        build_document_index([]),
    )

    assert rows[0]["decision"] == "reject"
    assert rows[0]["decision_reason"] == "generic_domain_word"


def test_build_entity_normalization_artifacts_writes_review_files(tmp_path: Path) -> None:
    source_dir = tmp_path / "rag"
    run_dir = tmp_path / "runs" / "ner"
    write_jsonl(
        source_dir / DOCUMENTS_FILE,
        [
            {
                "source": "osu-wiki",
                "page_id": "Beatmap/Drain_time",
                "title": "Drain time",
                "repo_rel_path": "Beatmap/Drain_time/en.md",
            }
        ],
    )
    write_jsonl(
        run_dir / ENTITY_CANDIDATES_FILE,
        [candidate("Drain time", "Beatmap/Drain_time", score=0.91)],
    )

    report = build_entity_normalization_artifacts(run_dir, source_artifact_dir=source_dir, output_dir=run_dir)

    assert report["by_decision"]["accept"] == 1
    assert (run_dir / ENTITY_NORMALIZATION_REVIEW_FILE).exists()


def candidate(text: str, document_id: str, *, score: float) -> dict:
    return {
        "text": text,
        "label": "osu gameplay mechanic or game mode term",
        "score": score,
        "document_id": document_id,
        "chunk_id": f"{document_id}::article",
        "context": f"{text} context",
    }
