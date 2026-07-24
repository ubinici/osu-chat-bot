from pathlib import Path

from osu_chatbot.config import load_config


def test_load_config_supports_artifact_and_qdrant_environment_overrides(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[artifacts]
path = "artifacts/rag"
source_path = "artifacts/source"

[qdrant]
url = "http://old-qdrant:6333"
collection = "old_collection"
vector_size = 384

[retrieval]
top_k = 3
alias_minimum_confidence = 0.9
alias_minimum_tokens = 3
preferred_document_limit = 2
soft_preferred_document_limit = 1
canonical_source_types = ["wiki", "handbook"]
temporal_source_types = ["news", "forum"]
excluded_chunk_types = ["citation"]
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("OSU_BOT_ARTIFACT_PATH", "artifacts/runs/test/rag")
    monkeypatch.setenv("OSU_BOT_ARTIFACT_SOURCE_PATH", "artifacts/rag")
    monkeypatch.setenv("OSU_BOT_QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("OSU_BOT_QDRANT_COLLECTION", "osu_wiki_en_test")
    monkeypatch.setenv("OSU_BOT_QDRANT_VECTOR_SIZE", "768")
    monkeypatch.setenv("OSU_BOT_GENERATION_PROVIDER", "openai-compatible")
    monkeypatch.setenv("OSU_BOT_GENERATION_URL", "https://models.example/v1/")
    monkeypatch.setenv("OSU_BOT_GENERATION_MODEL", "small-instruct")
    monkeypatch.setenv("OSU_BOT_GENERATION_API_KEY", "test-key")
    monkeypatch.setenv("OSU_BOT_GENERATION_THINK", "low")
    monkeypatch.setenv("OSU_BOT_STYLE_PROFILE_PATH", "artifacts/style/profile.json")
    monkeypatch.setenv("OSU_BOT_MAX_CONCURRENT_REQUESTS", "4")
    monkeypatch.setenv("OSU_BOT_DISCORD_TOKEN", "discord-secret")
    monkeypatch.setenv("OSU_BOT_DISCORD_API_URL", "http://app:8000/")
    monkeypatch.setenv("OSU_BOT_DISCORD_GUILD_ID", "12345")
    monkeypatch.setenv("OSU_BOT_DISCORD_FEEDBACK_PATH", "artifacts/feedback/discord.jsonl")
    monkeypatch.setenv("OSU_BOT_DISCORD_EPHEMERAL", "true")
    monkeypatch.setenv("OSU_BOT_DISCORD_MAX_ACTIVE_ROOMS", "6")
    monkeypatch.setenv("OSU_BOT_DISCORD_INACTIVITY_SECONDS", "420")
    monkeypatch.setenv("OSU_BOT_DISCORD_CONTEXT_TURNS", "5")
    monkeypatch.setenv(
        "OSU_BOT_DISCORD_SESSION_DB_PATH",
        "artifacts/feedback/sessions.sqlite3",
    )
    monkeypatch.setenv("OSU_BOT_DISCORD_CATEGORY_ID", "67890")

    config = load_config(config_path)

    assert config.artifacts.path == Path("artifacts/runs/test/rag")
    assert config.artifacts.source_path == Path("artifacts/rag")
    assert config.qdrant.url == "http://qdrant:6333"
    assert config.qdrant.collection == "osu_wiki_en_test"
    assert config.qdrant.vector_size == 768
    assert config.retrieval.top_k == 3
    assert config.retrieval.alias_minimum_confidence == 0.9
    assert config.retrieval.alias_minimum_tokens == 3
    assert config.retrieval.preferred_document_limit == 2
    assert config.retrieval.soft_preferred_document_limit == 1
    assert config.retrieval.canonical_source_types == ("wiki", "handbook")
    assert config.retrieval.temporal_source_types == ("news", "forum")
    assert config.retrieval.excluded_chunk_types == ("citation",)
    assert config.generation.provider == "openai-compatible"
    assert config.generation.url == "https://models.example/v1"
    assert config.generation.model == "small-instruct"
    assert config.generation.api_key == "test-key"
    assert config.generation.think == "low"
    assert config.generation.style_profile_path == Path("artifacts/style/profile.json")
    assert config.generation.max_concurrent_requests == 4
    assert config.discord.token == "discord-secret"
    assert config.discord.api_url == "http://app:8000"
    assert config.discord.guild_id == 12345
    assert config.discord.feedback_path == Path("artifacts/feedback/discord.jsonl")
    assert config.discord.ephemeral_answers is True
    assert config.discord.max_active_rooms == 6
    assert config.discord.inactivity_seconds == 420
    assert config.discord.context_turns == 5
    assert config.discord.session_db_path == Path("artifacts/feedback/sessions.sqlite3")
    assert config.discord.category_id == 67890
