from types import SimpleNamespace

from osu_chatbot.config import AppConfig, QdrantConfig, RetrievalConfig
from osu_chatbot.domain.models import Chunk, QueryIntent, SearchResult
from osu_chatbot.retrieval.analysis import ArtifactTopicResolver, DefaultQueryAnalyzer
from osu_chatbot.retrieval.dense import DenseRetriever, build_qdrant_filter, chunk_from_payload
from osu_chatbot.retrieval.models import QueryAnalysis, ResolvedTopic, RetrievalRequest
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
        RetrievalRequest(query="What is a beatmap?", limit=2),
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


def test_dense_filter_supports_source_document_and_chunk_policy() -> None:
    query_filter = build_qdrant_filter(
        RetrievalRequest(
            query="query",
            limit=3,
            source_types=("wiki",),
            document_ids=("Beatmap",),
            excluded_chunk_types=("formula", "citation"),
        )
    )

    assert query_filter is not None
    assert [(condition.key, condition.match.any) for condition in query_filter.must] == [
        ("source_type", ["wiki"]),
        ("document_id", ["Beatmap"]),
    ]
    assert [(condition.key, condition.match.any) for condition in query_filter.must_not] == [
        ("chunk_type", ["formula", "citation"]),
    ]


def test_retriever_expands_osu_aliases_and_delegates_to_backend() -> None:
    class FakeBackend:
        def __init__(self):
            self.calls = []

        def search(self, request: RetrievalRequest):
            self.calls.append(request)
            return [SearchResult(chunk=chunk(), score=0.8)]

    backend = FakeBackend()
    config = AppConfig(retrieval=RetrievalConfig(top_k=4))
    retriever = Retriever(config, backend=backend, analyzer=DefaultQueryAnalyzer())

    outcome = retriever.retrieve("what does AR change?")
    results = outcome.results

    assert results[0].chunk.document_id == "Beatmap"
    assert backend.calls == [
        RetrievalRequest(
            query="what does AR change?\nRelated osu! terms: approach, rate",
            limit=4,
            source_types=("wiki",),
            excluded_chunk_types=("citation", "formula"),
        )
    ]
    assert outcome.intent.labels == {"definition"}
    assert outcome.search_query.endswith("Related osu! terms: approach, rate")


def test_retriever_prefers_resolved_documents_then_fills_from_source_lane() -> None:
    preferred = SearchResult(chunk=chunk("preferred"), score=0.7)
    fallback = SearchResult(chunk=chunk("fallback"), score=0.9)

    class StaticAnalyzer:
        def analyze(self, query: str):
            return QueryAnalysis(
                query=query,
                intent=QueryIntent(labels={"definition"}),
                topics=(
                    ResolvedTopic(
                        canonical_id="Beatmap",
                        matched_alias="beatmap",
                        document_ids=("Beatmap",),
                        confidence=1.0,
                    ),
                ),
            )

    class FakeBackend:
        def __init__(self):
            self.calls = []

        def search(self, request: RetrievalRequest):
            self.calls.append(request)
            return [preferred] if request.document_ids else [fallback]

    backend = FakeBackend()
    config = AppConfig(retrieval=RetrievalConfig(top_k=2))
    outcome = Retriever(config, backend=backend, analyzer=StaticAnalyzer()).retrieve("What is a beatmap?")

    assert [result.chunk.id for result in outcome.results] == ["preferred", "fallback"]
    assert backend.calls[0].document_ids == ("Beatmap",)
    assert backend.calls[1].source_types == ("wiki",)


def test_artifact_topic_resolver_rejects_ambiguous_aliases(tmp_path) -> None:
    artifact = tmp_path / "aliases.jsonl"
    artifact.write_text(
        '{"alias":"stack leniency","document_id":"Beatmap/Stack_leniency",'
        '"canonical_document_id":"Beatmap/Stack_leniency","confidence":1.0}\n'
        '{"alias":"player rank","document_id":"Ranking","confidence":0.95}\n'
        '{"alias":"player rank","document_id":"Beatmap/Category","confidence":0.9}\n',
        encoding="utf-8",
    )

    resolver = ArtifactTopicResolver(artifact)

    assert resolver.resolve("What is stack leniency?")[0].canonical_id == "Beatmap/Stack_leniency"
    assert resolver.resolve("What is player rank?") == ()


def test_retriever_does_not_call_backend_for_blank_query() -> None:
    class FailBackend:
        def search(self, request: RetrievalRequest):
            raise AssertionError("backend should not be called")

    assert Retriever(AppConfig(), backend=FailBackend(), analyzer=DefaultQueryAnalyzer()).search("   ") == []


def test_retriever_delegates_readiness_to_backend() -> None:
    class ReadyBackend:
        def search(self, request: RetrievalRequest):
            return []

        def is_ready(self):
            return False

    assert not Retriever(AppConfig(), backend=ReadyBackend(), analyzer=DefaultQueryAnalyzer()).is_ready()
