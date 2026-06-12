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
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("OSU_BOT_ARTIFACT_PATH", "artifacts/runs/test/rag")
    monkeypatch.setenv("OSU_BOT_ARTIFACT_SOURCE_PATH", "artifacts/rag")
    monkeypatch.setenv("OSU_BOT_QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("OSU_BOT_QDRANT_COLLECTION", "osu_wiki_en_test")
    monkeypatch.setenv("OSU_BOT_QDRANT_VECTOR_SIZE", "768")

    config = load_config(config_path)

    assert config.artifacts.path == Path("artifacts/runs/test/rag")
    assert config.artifacts.source_path == Path("artifacts/rag")
    assert config.qdrant.url == "http://qdrant:6333"
    assert config.qdrant.collection == "osu_wiki_en_test"
    assert config.qdrant.vector_size == 768
