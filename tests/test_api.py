from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock

from fastapi.testclient import TestClient

from osu_chatbot.app.api import create_app
from osu_chatbot.app.service import ChatResult, SourceCitation


class FakeChatService:
    def __init__(self, *, ready: bool = True):
        self.ready = ready

    def is_ready(self) -> bool:
        return self.ready

    def ask(self, question: str) -> ChatResult:
        if not question.strip():
            raise ValueError("Question must not be blank.")
        return ChatResult(
            answer="AR controls object visibility time. [1]",
            sources=[
                SourceCitation(
                    citation=1,
                    chunk_id="ar::article",
                    document_id="Beatmap/Approach_rate",
                    title="Approach rate",
                    url="https://osu.ppy.sh/wiki/en/Beatmap/Approach_rate",
                    score=0.91,
                )
            ],
            intent=["definition"],
            search_query=question,
            latency_ms=12,
        )


def test_api_exposes_health_readiness_and_chat() -> None:
    client = TestClient(create_app(service=FakeChatService()))

    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json() == {"status": "ready"}

    response = client.post("/v1/chat", json={"question": "What does AR do?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].endswith("[1]")
    assert body["sources"][0]["document_id"] == "Beatmap/Approach_rate"
    assert body["retrieval_lane"] == "canonical"
    assert body["resolved_topics"] == []
    assert body["response_type"] == "answer"
    assert body["clarification"] is None
    assert body["latency_ms"] == 12


def test_api_reports_unready_index() -> None:
    client = TestClient(create_app(service=FakeChatService(ready=False)))

    response = client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["detail"] == "Retrieval index is not ready."


def test_api_rejects_blank_question() -> None:
    client = TestClient(create_app(service=FakeChatService()))

    response = client.post("/v1/chat", json={"question": "   "})

    assert response.status_code == 400


def test_async_chat_route_allows_overlapping_service_calls() -> None:
    class ConcurrentChatService(FakeChatService):
        def __init__(self):
            super().__init__()
            self.barrier = Barrier(2, timeout=2)
            self.lock = Lock()
            self.active = 0
            self.peak = 0

        def ask(self, question: str) -> ChatResult:
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            try:
                self.barrier.wait()
                return super().ask(question)
            finally:
                with self.lock:
                    self.active -= 1

    service = ConcurrentChatService()
    client = TestClient(create_app(service=service))

    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(
            pool.map(
                lambda question: client.post("/v1/chat", json={"question": question}),
                ["What does AR do?", "What does OD do?"],
            )
        )

    assert [response.status_code for response in responses] == [200, 200]
    assert service.peak == 2
