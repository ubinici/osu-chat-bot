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


def document_chunk(chunk_id: str, document_id: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id=document_id,
        source_type="wiki",
        file_path=f"{document_id}/en.md",
        osu_url=f"https://osu.ppy.sh/wiki/en/{document_id}",
        title=document_id.rsplit("/", 1)[-1].replace("_", " "),
        text=f"Evidence from {document_id}.",
        chunk_index=0,
        metadata={"chunk_type": "article"},
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


def test_retriever_limits_soft_topic_focus_before_general_retrieval() -> None:
    client = SearchResult(
        chunk=document_chunk("client::article", "Client"),
        score=0.7,
    )
    playfield = SearchResult(
        chunk=document_chunk("playfield::article", "Client/Playfield"),
        score=0.9,
    )
    interface = SearchResult(
        chunk=document_chunk("interface::article", "Client/Interface"),
        score=0.8,
    )

    class StaticAnalyzer:
        def analyze(self, query: str):
            return QueryAnalysis(
                query=query,
                topics=(
                    ResolvedTopic(
                        canonical_id="Client",
                        matched_alias="game client",
                        document_ids=("Client",),
                        confidence=0.96,
                        preference_strength="soft",
                    ),
                ),
            )

    class FakeBackend:
        def __init__(self):
            self.calls = []

        def search(self, request: RetrievalRequest):
            self.calls.append(request)
            return [client] if request.document_ids else [playfield, interface]

    backend = FakeBackend()
    config = AppConfig(
        retrieval=RetrievalConfig(
            top_k=3,
            preferred_document_limit=3,
            soft_preferred_document_limit=1,
        )
    )
    outcome = Retriever(config, backend=backend, analyzer=StaticAnalyzer()).retrieve(
        "Explain the buttons in the game client"
    )

    assert [result.chunk.document_id for result in outcome.results] == [
        "Client",
        "Client/Playfield",
        "Client/Interface",
    ]
    assert backend.calls[0].document_ids == ("Client",)
    assert backend.calls[0].limit == 1
    assert backend.calls[1].source_types == ("wiki",)
    assert backend.calls[1].limit == 3


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


def test_artifact_topic_resolver_softens_broad_aliases_but_keeps_specific_aliases_strong(
    tmp_path,
) -> None:
    artifact = tmp_path / "aliases.jsonl"
    artifact.write_text(
        '{"alias":"game client","document_id":"Client",'
        '"canonical_document_id":"Client","source":"accepted_link","confidence":0.96}\n'
        '{"alias":"HP drain","document_id":"Beatmap/HP_drain_rate",'
        '"source":"accepted_link","confidence":0.89}\n',
        encoding="utf-8",
    )

    resolver = ArtifactTopicResolver(artifact)

    assert resolver.resolve("buttons in the game client")[0].preference_strength == "soft"
    assert resolver.resolve("what is HP drain?")[0].preference_strength == "strong"


def test_retriever_does_not_call_backend_for_blank_query() -> None:
    class FailBackend:
        def search(self, request: RetrievalRequest):
            raise AssertionError("backend should not be called")

    assert Retriever(AppConfig(), backend=FailBackend(), analyzer=DefaultQueryAnalyzer()).search("   ") == []


def test_retriever_returns_clarification_without_searching_for_vague_query() -> None:
    class FailBackend:
        def search(self, request: RetrievalRequest):
            raise AssertionError("backend should not be called for a clarification")

    outcome = Retriever(
        AppConfig(),
        backend=FailBackend(),
        analyzer=DefaultQueryAnalyzer(),
    ).retrieve("it does not work")

    assert outcome.results == []
    assert outcome.search_query == "it does not work"
    assert outcome.analysis.requires_clarification
    assert outcome.analysis.clarification.reason == "missing_topic"


def test_clarification_policy_does_not_block_short_but_specific_queries() -> None:
    analysis = DefaultQueryAnalyzer().analyze("who created the game?")

    assert not analysis.requires_clarification


def test_retriever_delegates_readiness_to_backend() -> None:
    class ReadyBackend:
        def search(self, request: RetrievalRequest):
            return []

        def is_ready(self):
            return False

    assert not Retriever(AppConfig(), backend=ReadyBackend(), analyzer=DefaultQueryAnalyzer()).is_ready()
