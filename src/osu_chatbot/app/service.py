from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from time import perf_counter

from ..config import AppConfig
from ..generation import TextGenerator, create_generator
from ..generation.answerer import answer_question
from ..retrieval.service import Retriever


@dataclass(frozen=True)
class SourceCitation:
    citation: int
    chunk_id: str
    document_id: str
    title: str
    url: str
    score: float


@dataclass(frozen=True)
class TopicMatch:
    canonical_id: str
    matched_alias: str
    document_ids: list[str]
    confidence: float


@dataclass(frozen=True)
class ChatResult:
    answer: str
    sources: list[SourceCitation]
    intent: list[str]
    search_query: str
    latency_ms: int
    retrieval_lane: str = "canonical"
    resolved_topics: list[TopicMatch] | None = None


class ChatService:
    """Compose retrieval and generation while limiting CPU inference concurrency."""

    def __init__(
        self,
        config: AppConfig,
        *,
        retriever: Retriever | None = None,
        generator: TextGenerator | None = None,
    ):
        self.retriever = retriever or Retriever(config)
        self.generator = generator or create_generator(config.generation)
        self._inference_lock = Lock()

    def ask(self, question: str) -> ChatResult:
        clean_question = " ".join(question.split())
        if not clean_question:
            raise ValueError("Question must not be blank.")

        started = perf_counter()
        with self._inference_lock:
            outcome = self.retriever.retrieve(clean_question)
            answer = answer_question(clean_question, outcome.results, self.generator)
        latency_ms = round((perf_counter() - started) * 1000)

        sources = [
            SourceCitation(
                citation=index,
                chunk_id=result.chunk.id,
                document_id=result.chunk.document_id,
                title=result.chunk.title,
                url=result.chunk.osu_url,
                score=round(result.score, 6),
            )
            for index, result in enumerate(outcome.results, start=1)
        ]
        return ChatResult(
            answer=answer,
            sources=sources,
            intent=sorted(outcome.intent.labels),
            search_query=outcome.search_query,
            latency_ms=latency_ms,
            retrieval_lane=outcome.analysis.retrieval_lane,
            resolved_topics=[
                TopicMatch(
                    canonical_id=topic.canonical_id,
                    matched_alias=topic.matched_alias,
                    document_ids=list(topic.document_ids),
                    confidence=round(topic.confidence, 6),
                )
                for topic in outcome.analysis.topics
            ],
        )

    def is_ready(self) -> bool:
        return self.retriever.is_ready()
