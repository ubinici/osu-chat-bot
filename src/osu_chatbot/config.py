from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class CorpusConfig:
    osu_wiki_path: Path = Path("database/osu-wiki")
    language: str = "en"
    include_news: bool = True


@dataclass(frozen=True)
class ArtifactConfig:
    path: Path = Path("artifacts/rag")
    source_path: Path | None = None


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cache_folder: str | None = None
    device: str | None = None
    local_files_only: bool = False


@dataclass(frozen=True)
class QdrantConfig:
    url: str = "file://artifacts/rag/qdrant"
    collection: str = "osu_wiki_en_dense_v2"
    vector_size: int = 384


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 6
    alias_artifact: str = "document_aliases.jsonl"
    alias_minimum_confidence: float = 0.85
    alias_minimum_tokens: int = 2
    preferred_document_limit: int = 4
    soft_preferred_document_limit: int = 1
    canonical_source_types: tuple[str, ...] = ("wiki",)
    temporal_source_types: tuple[str, ...] = ("wiki", "news")
    excluded_chunk_types: tuple[str, ...] = ("citation", "formula")


@dataclass(frozen=True)
class GenerationConfig:
    provider: str = "ollama"
    url: str = "http://127.0.0.1:11434"
    model: str = "mistral"
    api_key: str | None = None
    think: bool | str = False
    temperature: float = 0.1
    timeout_seconds: float = 120.0
    style_profile_path: Path | None = None
    max_concurrent_requests: int = 1


@dataclass(frozen=True)
class DiscordConfig:
    token: str | None = None
    api_url: str = "http://127.0.0.1:8000"
    guild_id: int | None = None
    feedback_path: Path = Path("artifacts/feedback/events.jsonl")
    answer_version: str = "gpt-oss-discord-v1"
    ephemeral_answers: bool = False


@dataclass(frozen=True)
class AppConfig:
    corpus: CorpusConfig = CorpusConfig()
    artifacts: ArtifactConfig = ArtifactConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    qdrant: QdrantConfig = QdrantConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    generation: GenerationConfig = GenerationConfig()
    discord: DiscordConfig = DiscordConfig()


def load_config(path: str | Path = "config.toml") -> AppConfig:
    config_path = Path(path)
    data = tomllib.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}

    artifact_data = data.get("artifacts", {})
    source_path = os.environ.get("OSU_BOT_ARTIFACT_SOURCE_PATH") or artifact_data.get("source_path")
    qdrant_data = data.get("qdrant", {})
    retrieval_data = data.get("retrieval", {})
    generation_data = data.get("generation", {})
    discord_data = data.get("discord", {})
    if not generation_data and data.get("ollama"):
        generation_data = {"provider": "ollama", **data["ollama"]}
    return AppConfig(
        corpus=CorpusConfig(
            osu_wiki_path=Path(data.get("corpus", {}).get("osu_wiki_path", "database/osu-wiki")),
            language=data.get("corpus", {}).get("language", "en"),
            include_news=bool(data.get("corpus", {}).get("include_news", True)),
        ),
        artifacts=ArtifactConfig(
            path=Path(os.environ.get("OSU_BOT_ARTIFACT_PATH") or artifact_data.get("path", "artifacts/rag")),
            source_path=Path(source_path) if source_path else None,
        ),
        embedding=EmbeddingConfig(
            model=data.get("embedding", {}).get("model", "sentence-transformers/all-MiniLM-L6-v2"),
            cache_folder=data.get("embedding", {}).get("cache_folder"),
            device=data.get("embedding", {}).get("device"),
            local_files_only=bool(data.get("embedding", {}).get("local_files_only", False)),
        ),
        qdrant=QdrantConfig(
            url=os.environ.get("OSU_BOT_QDRANT_URL") or qdrant_data.get("url", "file://artifacts/rag/qdrant"),
            collection=os.environ.get("OSU_BOT_QDRANT_COLLECTION") or qdrant_data.get("collection", "osu_wiki_en_dense_v2"),
            vector_size=int(os.environ.get("OSU_BOT_QDRANT_VECTOR_SIZE") or qdrant_data.get("vector_size", 384)),
        ),
        retrieval=RetrievalConfig(
            top_k=int(retrieval_data.get("top_k", 6)),
            alias_artifact=str(retrieval_data.get("alias_artifact", "document_aliases.jsonl")),
            alias_minimum_confidence=float(retrieval_data.get("alias_minimum_confidence", 0.85)),
            alias_minimum_tokens=max(1, int(retrieval_data.get("alias_minimum_tokens", 2))),
            preferred_document_limit=max(1, int(retrieval_data.get("preferred_document_limit", 4))),
            soft_preferred_document_limit=max(
                1, int(retrieval_data.get("soft_preferred_document_limit", 1))
            ),
            canonical_source_types=_string_tuple(
                retrieval_data.get("canonical_source_types"),
                ("wiki",),
            ),
            temporal_source_types=_string_tuple(
                retrieval_data.get("temporal_source_types"),
                ("wiki", "news"),
            ),
            excluded_chunk_types=_string_tuple(
                retrieval_data.get("excluded_chunk_types"),
                ("citation", "formula"),
            ),
        ),
        generation=GenerationConfig(
            provider=(
                os.environ.get("OSU_BOT_GENERATION_PROVIDER")
                or generation_data.get("provider", "ollama")
            ).strip().casefold(),
            url=(
                os.environ.get("OSU_BOT_GENERATION_URL")
                or generation_data.get("url", "http://127.0.0.1:11434")
            ).rstrip("/"),
            model=os.environ.get("OSU_BOT_GENERATION_MODEL") or generation_data.get("model", "mistral"),
            api_key=os.environ.get("OSU_BOT_GENERATION_API_KEY") or generation_data.get("api_key"),
            think=_parse_think(
                os.environ.get("OSU_BOT_GENERATION_THINK")
                if "OSU_BOT_GENERATION_THINK" in os.environ
                else generation_data.get("think", False)
            ),
            temperature=float(
                os.environ.get("OSU_BOT_GENERATION_TEMPERATURE")
                or generation_data.get("temperature", 0.1)
            ),
            timeout_seconds=float(
                os.environ.get("OSU_BOT_GENERATION_TIMEOUT_SECONDS")
                or generation_data.get("timeout_seconds", 120.0)
            ),
            style_profile_path=_optional_path(
                os.environ.get("OSU_BOT_STYLE_PROFILE_PATH")
                or generation_data.get("style_profile_path")
            ),
            max_concurrent_requests=max(
                1,
                int(
                    os.environ.get("OSU_BOT_MAX_CONCURRENT_REQUESTS")
                    or generation_data.get("max_concurrent_requests", 1)
                ),
            ),
        ),
        discord=DiscordConfig(
            token=os.environ.get("OSU_BOT_DISCORD_TOKEN") or discord_data.get("token"),
            api_url=(
                os.environ.get("OSU_BOT_DISCORD_API_URL")
                or discord_data.get("api_url", "http://127.0.0.1:8000")
            ).rstrip("/"),
            guild_id=_optional_int(
                os.environ.get("OSU_BOT_DISCORD_GUILD_ID")
                or discord_data.get("guild_id")
            ),
            feedback_path=Path(
                os.environ.get("OSU_BOT_DISCORD_FEEDBACK_PATH")
                or discord_data.get("feedback_path", "artifacts/feedback/events.jsonl")
            ),
            answer_version=str(
                os.environ.get("OSU_BOT_ANSWER_VERSION")
                or discord_data.get("answer_version", "gpt-oss-discord-v1")
            ).strip(),
            ephemeral_answers=_parse_bool(
                os.environ.get("OSU_BOT_DISCORD_EPHEMERAL")
                if "OSU_BOT_DISCORD_EPHEMERAL" in os.environ
                else discord_data.get("ephemeral_answers", False),
                label="discord ephemeral_answers",
            ),
        ),
    )


def _parse_think(value: object) -> bool | str:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"low", "medium", "high"}:
        return normalized
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"", "0", "false", "no", "off"}:
        return False
    raise ValueError("generation think must be true, false, low, medium, or high")


def _parse_bool(value: object, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"", "0", "false", "no", "off"}:
        return False
    raise ValueError(f"{label} must be true or false")


def _optional_int(value: object) -> int | None:
    if value is None or not str(value).strip():
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("discord guild_id must be a positive integer")
    return parsed


def _optional_path(value: object) -> Path | None:
    if value is None or not str(value).strip():
        return None
    return Path(str(value).strip())


def _string_tuple(value: object, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if not isinstance(value, (list, tuple)):
        raise ValueError("retrieval source and chunk type settings must be arrays")
    return tuple(str(item).strip() for item in value if str(item).strip())
