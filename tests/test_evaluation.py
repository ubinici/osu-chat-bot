from pathlib import Path

from osu_chatbot.config import AppConfig, ArtifactConfig, RetrievalConfig
from osu_chatbot.domain.artifacts import write_json, write_jsonl
from osu_chatbot.domain.models import Chunk
from osu_chatbot.evaluation.runner import run_evaluation


def test_run_evaluation_scores_keyword_retrieval_without_dense_index(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "rag"
    dataset = tmp_path / "questions.jsonl"
    chunks = [
        Chunk(
            id="beatmap::article",
            document_id="Beatmap",
            source_type="wiki",
            file_path="Beatmap/en.md",
            osu_url="https://osu.ppy.sh/wiki/en/Beatmap",
            title="Beatmap",
            text="A beatmap is a set of game levels.",
            chunk_index=0,
            heading_path=["Beatmap"],
            metadata={"chunk_type": "article"},
        )
    ]
    write_jsonl(artifact_dir / "chunks_hierarchical.jsonl", chunks)
    write_jsonl(artifact_dir / "documents_structured.jsonl", [])
    write_json(artifact_dir / "terms.json", [])
    dataset.write_text(
        '{"category": "definition", "question": "What is a beatmap?", "expected_document_ids": ["Beatmap"]}\n',
        encoding="utf-8",
    )
    config = AppConfig(artifacts=ArtifactConfig(path=artifact_dir), retrieval=RetrievalConfig(final_top_k=1))

    report = run_evaluation(config, dataset, use_dense=False)

    assert report["summary"]["examples"] == 1
    assert report["summary"]["matches"] == 1
    assert report["by_category"]["definition"]["matches"] == 1
    assert report["examples"][0]["category"] == "definition"
    assert report["examples"][0]["expected_document_ids"] == ["Beatmap"]
    assert report["examples"][0]["top_sources"][0]["document_id"] == "Beatmap"
