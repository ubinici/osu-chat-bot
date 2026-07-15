from osu_chatbot.app.service import ChatService
from osu_chatbot.config import AppConfig
from osu_chatbot.domain.models import Chunk, QueryIntent, SearchResult
from osu_chatbot.retrieval.models import QueryAnalysis
from osu_chatbot.retrieval.service import RetrievalOutcome


def test_chat_service_runs_complete_cited_pipeline() -> None:
    search_result = SearchResult(
        chunk=Chunk(
            id="ar::article",
            document_id="Beatmap/Approach_rate",
            source_type="wiki",
            file_path="Beatmap/Approach_rate/en.md",
            osu_url="https://osu.ppy.sh/wiki/en/Beatmap/Approach_rate",
            title="Approach rate",
            text="Approach rate controls how long hit objects are visible.",
            chunk_index=0,
        ),
        score=0.91234567,
    )

    class FakeRetriever:
        def retrieve(self, question: str):
            assert question == "what does AR do?"
            return RetrievalOutcome(
                results=[search_result],
                analysis=QueryAnalysis(
                    query=question,
                    intent=QueryIntent(labels={"definition"}),
                ),
                search_query="what does AR do?\nRelated osu! terms: approach, rate",
            )

        def is_ready(self):
            return True

    class FakeGenerator:
        def generate(self, prompt: str):
            assert "Approach rate controls" in prompt
            return "AR changes how long objects are visible. [1]"

    service = ChatService(
        AppConfig(),
        retriever=FakeRetriever(),
        generator=FakeGenerator(),
    )

    response = service.ask("  what does AR do?  ")

    assert response.answer.endswith("[1]")
    assert response.intent == ["definition"]
    assert response.sources[0].citation == 1
    assert response.sources[0].score == 0.912346
    assert response.retrieval_lane == "canonical"
    assert response.resolved_topics == []
    assert response.latency_ms >= 0
    assert service.is_ready()
