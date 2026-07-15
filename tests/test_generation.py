from osu_chatbot.domain.models import Chunk, SearchResult
from osu_chatbot.generation.answerer import INSUFFICIENT_CONTEXT, answer_question
from osu_chatbot.generation.factory import create_generator
from osu_chatbot.generation.ollama import OllamaGenerator
from osu_chatbot.generation.openai_compatible import OpenAICompatibleGenerator
from osu_chatbot.generation.prompt import build_prompt
from osu_chatbot.config import GenerationConfig


def test_prompt_requests_grounded_conversational_answer_with_citations() -> None:
    result = SearchResult(
        chunk=Chunk(
            id="ar::article",
            document_id="Beatmap/Approach_rate",
            source_type="wiki",
            file_path="Beatmap/Approach_rate/en.md",
            osu_url="https://osu.ppy.sh/wiki/en/Beatmap/Approach_rate",
            title="Approach rate",
            text="Approach rate controls how long hit objects are visible.",
            chunk_index=0,
            heading_path=["Approach rate"],
        ),
        score=0.9,
    )

    prompt = build_prompt("what does AR do?", [result])

    assert "natural, conversational tone" in prompt
    assert "using only" not in prompt
    assert "Use only the cited" in prompt
    assert "[1]" in prompt
    assert result.chunk.osu_url in prompt


def test_answerer_uses_injected_generator() -> None:
    class FakeGenerator:
        def __init__(self):
            self.prompts = []

        def generate(self, prompt: str) -> str:
            self.prompts.append(prompt)
            return "AR controls object visibility time. [1]"

    generator = FakeGenerator()
    result = SearchResult(
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
        score=0.9,
    )

    answer = answer_question("what does AR do?", [result], generator)

    assert answer.endswith("[1]")
    assert len(generator.prompts) == 1


def test_answerer_skips_generation_without_context() -> None:
    class FailGenerator:
        def generate(self, prompt: str) -> str:
            raise AssertionError("generator should not be called")

    assert answer_question("unknown?", [], FailGenerator()) == INSUFFICIENT_CONTEXT


def test_generator_factory_selects_configured_provider() -> None:
    assert isinstance(create_generator(GenerationConfig(provider="ollama")), OllamaGenerator)
    assert isinstance(
        create_generator(GenerationConfig(provider="openai-compatible")),
        OpenAICompatibleGenerator,
    )


def test_openai_compatible_generator_sends_chat_completion(monkeypatch) -> None:
    captured = {}

    def fake_post_json(url, payload, **kwargs):
        captured.update(url=url, payload=payload, **kwargs)
        return {"choices": [{"message": {"content": "Grounded answer. [1]"}}]}

    monkeypatch.setattr("osu_chatbot.generation.openai_compatible.post_json", fake_post_json)
    config = GenerationConfig(
        provider="openai-compatible",
        url="https://models.example/v1",
        model="small-instruct",
        api_key="secret",
    )

    answer = OpenAICompatibleGenerator(config).generate("prompt")

    assert answer == "Grounded answer. [1]"
    assert captured["url"] == "https://models.example/v1/chat/completions"
    assert captured["headers"] == {"Authorization": "Bearer secret"}
    assert captured["payload"]["model"] == "small-instruct"


def test_ollama_generator_disables_thinking_for_cpu_latency(monkeypatch) -> None:
    captured = {}

    def fake_post_json(url, payload, **kwargs):
        captured.update(url=url, payload=payload, **kwargs)
        return {"response": "Grounded answer. [1]"}

    monkeypatch.setattr("osu_chatbot.generation.ollama.post_json", fake_post_json)
    config = GenerationConfig(provider="ollama", model="qwen3:4b")

    answer = OllamaGenerator(config).generate("prompt")

    assert answer == "Grounded answer. [1]"
    assert captured["url"] == "http://127.0.0.1:11434/api/generate"
    assert captured["payload"]["think"] is False
    assert captured["headers"] == {}


def test_ollama_generator_authenticates_cloud_requests(monkeypatch) -> None:
    captured = {}

    def fake_post_json(url, payload, **kwargs):
        captured.update(url=url, payload=payload, **kwargs)
        return {"response": "Cloud answer. [1]"}

    monkeypatch.setattr("osu_chatbot.generation.ollama.post_json", fake_post_json)
    config = GenerationConfig(
        provider="ollama",
        url="https://ollama.com",
        model="cloud-model",
        api_key="cloud-secret",
    )

    answer = OllamaGenerator(config).generate("prompt")

    assert answer == "Cloud answer. [1]"
    assert captured["url"] == "https://ollama.com/api/generate"
    assert captured["headers"] == {"Authorization": "Bearer cloud-secret"}
