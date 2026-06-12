from pathlib import Path

from osu_chatbot.domain.artifacts import write_json, write_jsonl
from osu_chatbot.domain.models import Chunk, Entity
from osu_chatbot.quality.stats import build_stats_report
from osu_chatbot.quality.validation import build_validation_report


def test_stats_report_summarizes_documents_chunks_and_terms(tmp_path: Path) -> None:
    artifact_dir = write_fixture_artifacts(tmp_path)

    report = build_stats_report(artifact_dir)

    assert report["documents"]["total"] == 2
    assert report["documents"]["by_source"] == {"osu-news": 1, "osu-wiki": 1}
    assert report["chunks"]["by_type"]["article"] == 1
    assert report["chunks"]["by_type"]["section"] == 1
    assert report["terms"]["total_entities"] == 1
    assert (artifact_dir / "stats_report.json").exists()


def test_validation_report_flags_database_quality_issues(tmp_path: Path) -> None:
    artifact_dir = write_fixture_artifacts(tmp_path, include_issues=True)

    report = build_validation_report(artifact_dir)
    kinds = {issue["kind"] for issue in report["issues"]}

    assert "duplicate_document_id" in kinds
    assert "invalid_news_date" in kinds
    assert "empty_chunk_text" in kinds
    assert "stopword_alias" in kinds
    assert report["summary"]["by_severity"]["error"] >= 2
    assert (artifact_dir / "validation_report.json").exists()


def write_fixture_artifacts(tmp_path: Path, include_issues: bool = False) -> Path:
    artifact_dir = tmp_path / "rag"
    documents = [
        {
            "source": "osu-wiki",
            "page_id": "Beatmap",
            "repo_rel_path": "Beatmap/en.md",
            "osu_url": "https://osu.ppy.sh/wiki/en/Beatmap",
            "title": "Beatmap",
            "domain": "beatmap",
            "subculture": "mapping",
            "sections": [{"section_id": "beatmap", "title": "Beatmap", "text": "A beatmap is a set of levels."}],
            "section_count": 1,
            "edge_count": 0,
            "citation_count": 0,
            "table_count": 0,
            "formula_count": 0,
            "image_count": 0,
        },
        {
            "source": "osu-news",
            "post_id": "news/2026/example",
            "repo_rel_path": "news/2026/example.md",
            "osu_url": "https://osu.ppy.sh/home/news/example",
            "title": "Example News",
            "domain": "news",
            "subculture": "community",
            "year": "2026",
            "series_primary": "community",
            "date_iso": "" if include_issues else "2026-06-07T20:00:00+00:00",
            "preview_paragraph": "Preview text.",
            "sections": [{"section_id": "intro", "title": "Intro", "text": "Preview text."}],
            "section_count": 1,
            "edge_count": 0,
            "citation_count": 0,
            "table_count": 0,
            "formula_count": 0,
            "image_count": 0,
        },
    ]
    if include_issues:
        documents.append(dict(documents[0]))

    chunks = [
        Chunk(
            id="beatmap::article",
            document_id="Beatmap",
            source_type="wiki",
            file_path="Beatmap/en.md",
            osu_url="https://osu.ppy.sh/wiki/en/Beatmap",
            title="Beatmap",
            text="A beatmap is a set of levels.",
            chunk_index=0,
            heading_path=["Beatmap"],
            metadata={"chunk_type": "article", "source": "osu-wiki"},
        ),
        Chunk(
            id="news::section",
            document_id="news/2026/example",
            source_type="news",
            file_path="news/2026/example.md",
            osu_url="https://osu.ppy.sh/home/news/example",
            title="Example News",
            text="" if include_issues else "Preview text.",
            chunk_index=1,
            heading_path=["Intro"],
            metadata={"chunk_type": "section", "source": "osu-news"},
        ),
    ]
    entities = [
        Entity(
            canonical="Beatmap",
            aliases=["Beatmap", "was"] if include_issues else ["Beatmap"],
            sources=["fixture"],
        )
    ]

    write_jsonl(artifact_dir / "documents_structured.jsonl", documents)
    write_jsonl(artifact_dir / "chunks_hierarchical.jsonl", chunks)
    write_json(artifact_dir / "terms.json", [entity.__dict__ for entity in entities])
    return artifact_dir
