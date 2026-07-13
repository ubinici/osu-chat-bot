from pathlib import Path

from osu_chatbot.config import AppConfig, RetrievalConfig
from osu_chatbot.domain.models import Chunk, SearchResult
from osu_chatbot.evaluation.runner import run_evaluation
from osu_chatbot.retrieval.service import Retriever


def test_run_evaluation_scores_dense_retrieval_and_reports_query_analysis(tmp_path: Path) -> None:
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(
        '{"category": "definition", "question": "What is a beatmap?", '
        '"expected_document_ids": ["Beatmap"]}\n',
        encoding="utf-8",
    )
    result = SearchResult(
        chunk=Chunk(
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
        ),
        score=0.93,
    )

    class StaticBackend:
        def search(self, query: str, limit: int):
            assert query == "What is a beatmap?"
            assert limit == 1
            return [result]

    config = AppConfig(retrieval=RetrievalConfig(top_k=1))
    retriever = Retriever(config, backend=StaticBackend())
    report = run_evaluation(config, dataset, retriever=retriever)

    assert report["summary"]["examples"] == 1
    assert report["summary"]["matches"] == 1
    assert report["by_category"]["definition"]["matches"] == 1
    assert report["examples"][0]["intent"] == ["definition"]
    assert report["examples"][0]["top_sources"] == [
        {
            "chunk_id": "beatmap::article",
            "document_id": "Beatmap",
            "title": "Beatmap",
            "score": 0.93,
        }
    ]


def test_evaluation_requires_actual_document_ids(tmp_path: Path) -> None:
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(
        '{"question": "What is HP drain?", "expected_document_ids": ["Beatmap/HP_drain_rate"]}\n',
        encoding="utf-8",
    )
    result = SearchResult(
        chunk=Chunk(
            id="hp::article",
            document_id="Beatmap/HP_drain_rate",
            source_type="wiki",
            file_path="Beatmap/HP_drain_rate/en.md",
            osu_url="https://osu.ppy.sh/wiki/en/Beatmap/HP_drain_rate",
            title="HP drain rate",
            text="HP drain rate controls passive health drain.",
            chunk_index=0,
        ),
        score=0.8,
    )

    class StaticBackend:
        def search(self, query: str, limit: int):
            return [result]

    config = AppConfig(retrieval=RetrievalConfig(top_k=1))
    report = run_evaluation(config, dataset, retriever=Retriever(config, backend=StaticBackend()))

    assert report["summary"]["matches"] == 1
    assert report["examples"][0]["document_match"] == 1
