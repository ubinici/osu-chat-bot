from types import SimpleNamespace

from osu_chatbot.config import AppConfig, QdrantConfig, RetrievalConfig
from osu_chatbot.domain.models import Chunk, SearchResult
from osu_chatbot.retrieval.dense import DenseRetriever, chunk_from_payload
from osu_chatbot.retrieval.service import Retriever


def chunk(chunk_id: str = "beatmap::article") -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="Beatmap",
        source_type="wiki",
        file_path="Beatmap/en.md",
        osu_url="https://osu.ppy.sh/wiki/en/Beatmap",
        title="Beatmap",
        text="A beatmap contains the hit objects used during play.",
        chunk_index=0,
        heading_path=["Beatmap"],
        metadata={"chunk_type": "article", "domain": "beatmap"},
    )


def test_dense_retriever_returns_self_contained_qdrant_payloads() -> None:
    class FakeEmbedder:
        def encode(self, texts):
            assert texts == ["What is a beatmap?"]
            return [[0.1, 0.2, 0.3]]

    class FakeClient:
        def query_points(self, **kwargs):
            assert kwargs == {
                "collection_name": "test_collection",
                "query": [0.1, 0.2, 0.3],
                "limit": 2,
                "with_payload": True,
            }
            return SimpleNamespace(
                points=[
                    SimpleNamespace(
                        score=0.91,
                        payload={
                            "chunk_id": "beatmap::article",
                            "document_id": "Beatmap",
                            "source_type": "wiki",
                            "file_path": "Beatmap/en.md",
                            "osu_url": "https://osu.ppy.sh/wiki/en/Beatmap",
                            "title": "Beatmap",
                            "text": "A beatmap contains hit objects.",
                            "chunk_index": 0,
                            "heading_path": ["Beatmap"],
                            "tags": ["mapping"],
                            "chunk_type": "article",
                            "domain": "beatmap",
                        },
                    )
                ]
            )

    config = AppConfig(qdrant=QdrantConfig(collection="test_collection"))
    results = DenseRetriever(config, embedder=FakeEmbedder(), client=FakeClient()).search(
        "What is a beatmap?",
        limit=2,
    )

    assert len(results) == 1
    assert results[0].score == 0.91
    assert results[0].chunk.document_id == "Beatmap"
    assert results[0].chunk.metadata == {"chunk_type": "article", "domain": "beatmap"}


def test_chunk_from_payload_rejects_incomplete_search_results() -> None:
    try:
        chunk_from_payload({"chunk_id": "missing-text", "document_id": "Beatmap"})
    except ValueError as exc:
        assert "missing" in str(exc).casefold()
    else:
        raise AssertionError("invalid payload should not become a chunk")


def test_retriever_expands_osu_aliases_and_delegates_to_backend() -> None:
    class FakeBackend:
        def __init__(self):
            self.calls = []

        def search(self, query: str, limit: int):
            self.calls.append((query, limit))
            return [SearchResult(chunk=chunk(), score=0.8)]

    backend = FakeBackend()
    config = AppConfig(retrieval=RetrievalConfig(top_k=4))
    retriever = Retriever(config, backend=backend)

    results = retriever.search("what does AR change?")

    assert results[0].chunk.document_id == "Beatmap"
    assert backend.calls == [("what does AR change?\nRelated osu! terms: approach, rate", 4)]
    assert retriever.last_intent.labels == set()


def test_retriever_does_not_call_backend_for_blank_query() -> None:
    class FailBackend:
        def search(self, query: str, limit: int):
            raise AssertionError("backend should not be called")

    assert Retriever(AppConfig(), backend=FailBackend()).search("   ") == []
