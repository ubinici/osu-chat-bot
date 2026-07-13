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


@dataclass(frozen=True)
class GenerationConfig:
    provider: str = "ollama"
    url: str = "http://127.0.0.1:11434"
    model: str = "mistral"
    api_key: str | None = None
    temperature: float = 0.1
    timeout_seconds: float = 120.0


@dataclass(frozen=True)
class AppConfig:
    corpus: CorpusConfig = CorpusConfig()
    artifacts: ArtifactConfig = ArtifactConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    qdrant: QdrantConfig = QdrantConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    generation: GenerationConfig = GenerationConfig()


def load_config(path: str | Path = "config.toml") -> AppConfig:
    config_path = Path(path)
    data = tomllib.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}

    artifact_data = data.get("artifacts", {})
    source_path = os.environ.get("OSU_BOT_ARTIFACT_SOURCE_PATH") or artifact_data.get("source_path")
    qdrant_data = data.get("qdrant", {})
    generation_data = data.get("generation", {})
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
            top_k=int(data.get("retrieval", {}).get("top_k", 6)),
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
            temperature=float(
                os.environ.get("OSU_BOT_GENERATION_TEMPERATURE")
                or generation_data.get("temperature", 0.1)
            ),
            timeout_seconds=float(
                os.environ.get("OSU_BOT_GENERATION_TIMEOUT_SECONDS")
                or generation_data.get("timeout_seconds", 120.0)
            ),
        ),
    )
