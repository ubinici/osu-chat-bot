from pathlib import Path

from osu_chatbot.config import AppConfig, RetrievalConfig
from osu_chatbot.domain.models import Chunk, SearchResult
from osu_chatbot.evaluation.datasets import EvaluationExample
from osu_chatbot.evaluation.metrics import score_retrieval
from osu_chatbot.evaluation.runner import run_evaluation
from osu_chatbot.retrieval.analysis import DefaultQueryAnalyzer
from osu_chatbot.retrieval.models import RetrievalRequest
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
        def search(self, request: RetrievalRequest):
            assert request.query == "What is a beatmap?"
            assert request.limit == 1
            return [result]

    config = AppConfig(retrieval=RetrievalConfig(top_k=1))
    retriever = Retriever(config, backend=StaticBackend(), analyzer=DefaultQueryAnalyzer())
    report = run_evaluation(config, dataset, retriever=retriever)

    assert report["summary"]["examples"] == 1
    assert report["summary"]["matches"] == 1
    assert report["summary"]["strict_matches"] == 1
    assert report["summary"]["acceptable_only_matches"] == 0
    assert report["summary"]["hit_at_1"] == 1.0
    assert report["summary"]["strict_hit_at_1"] == 1.0
    assert report["summary"]["mean_reciprocal_rank"] == 1.0
    assert report["by_category"]["definition"]["matches"] == 1
    assert report["examples"][0]["intent"] == ["definition"]
    assert report["examples"][0]["retrieval_lane"] == "canonical"
    assert report["examples"][0]["resolved_topics"] == []
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
        def search(self, request: RetrievalRequest):
            return [result]

    config = AppConfig(retrieval=RetrievalConfig(top_k=1))
    report = run_evaluation(
        config,
        dataset,
        retriever=Retriever(config, backend=StaticBackend(), analyzer=DefaultQueryAnalyzer()),
    )

    assert report["summary"]["matches"] == 1
    assert report["examples"][0]["document_match"] == 1
    assert report["examples"][0]["document_match_kind"] == "primary"


def test_evaluation_tracks_explicit_acceptable_documents_separately(tmp_path: Path) -> None:
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text(
        '{"question": "What files does osu install?", '
        '"expected_document_ids": ["Client/Installation"], '
        '"acceptable_document_ids": ["Client/Program_files"]}\n',
        encoding="utf-8",
    )
    result = _result_for_document(
        "Client/Program_files",
        title="osu! program files",
        text="The installation stores files in these directories.",
    )

    class StaticBackend:
        def search(self, request: RetrievalRequest):
            return [result]

    config = AppConfig(retrieval=RetrievalConfig(top_k=1))
    report = run_evaluation(
        config,
        dataset,
        retriever=Retriever(config, backend=StaticBackend(), analyzer=DefaultQueryAnalyzer()),
    )

    summary = report["summary"]
    row = report["examples"][0]
    assert summary["matches"] == 1
    assert summary["strict_matches"] == 0
    assert summary["acceptable_only_matches"] == 1
    assert summary["retrieval_accuracy"] == 1.0
    assert summary["strict_retrieval_accuracy"] == 0.0
    assert row["acceptable_document_ids"] == ["Client/Program_files"]
    assert row["primary_document_match"] == 0
    assert row["acceptable_document_match"] == 1
    assert row["matched_document_id"] == "Client/Program_files"
    assert row["document_match_kind"] == "acceptable"


def test_evaluation_does_not_infer_document_relationships_from_path_prefixes() -> None:
    result = _result_for_document(
        "History_of_osu!/2007",
        title="History of osu! 2007",
        text="The first public release happened in 2007.",
    )
    strict = score_retrieval(
        EvaluationExample(
            question="When did osu! start?",
            expected_document_ids=["History_of_osu!"],
        ),
        [result],
    )
    relation_aware = score_retrieval(
        EvaluationExample(
            question="When did osu! start?",
            expected_document_ids=["History_of_osu!"],
            acceptable_document_ids=["History_of_osu!/2007"],
        ),
        [result],
    )

    assert strict["matched"] == 0
    assert strict["document_match_kind"] == "none"
    assert relation_aware["matched"] == 1
    assert relation_aware["strict_matched"] == 0
    assert relation_aware["document_match_kind"] == "acceptable"


def _result_for_document(document_id: str, *, title: str, text: str) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            id=f"{document_id}::article",
            document_id=document_id,
            source_type="wiki",
            file_path=f"{document_id}/en.md",
            osu_url=f"https://osu.ppy.sh/wiki/en/{document_id}",
            title=title,
            text=text,
            chunk_index=0,
        ),
        score=0.8,
    )
