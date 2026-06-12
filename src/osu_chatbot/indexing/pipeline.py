from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import time

from ..config import AppConfig
from ..domain.artifacts import CHUNKS_FILE, INDEX_REPORT_FILE, INDEX_STATE_FILE, load_chunks, read_json, write_json
from ..domain.models import Chunk
from .embeddings import SentenceTransformerEmbeddings
from .vector_store import qdrant_client, qdrant_payload

STATE_FILE = INDEX_STATE_FILE
EMBEDDING_INPUT_VERSION = "metadata_text_v1"


@dataclass(frozen=True)
class IndexOptions:
    batch_size: int = 64
    offset: int = 0
    limit: int | None = None
    resume: bool = False


def build_index(
    config: AppConfig,
    options: IndexOptions | None = None,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    options = options or IndexOptions()
    if options.batch_size <= 0:
        raise RuntimeError("batch_size must be greater than zero.")
    if options.limit is not None and options.limit <= 0:
        raise RuntimeError("limit must be greater than zero when provided.")
    input_artifact_dir = config.artifacts.source_path or config.artifacts.path
    output_artifact_dir = config.artifacts.path
    chunk_artifact = input_artifact_dir / CHUNKS_FILE
    chunks = load_chunks(chunk_artifact)
    if not chunks:
        raise RuntimeError("No chunks found. Run `osu-bot ingest` first.")

    artifact_fingerprint = file_fingerprint(chunk_artifact)
    start_offset = resolve_start_offset(output_artifact_dir, options, artifact_fingerprint)
    end_offset = min(len(chunks), start_offset + options.limit) if options.limit else len(chunks)
    if start_offset >= len(chunks):
        raise RuntimeError(f"Index offset {start_offset} is past the available {len(chunks)} chunks.")
    if end_offset <= start_offset:
        raise RuntimeError("Index interval is empty. Check --offset and --limit.")

    emit(
        progress,
        {
            "event": "start",
            "total_chunks": len(chunks),
            "start_offset": start_offset,
            "end_offset": end_offset,
            "chunks_to_index": end_offset - start_offset,
            "batch_size": options.batch_size,
            "embedding_model": config.embedding.model,
            "collection": config.qdrant.collection,
        },
    )

    load_started = time.monotonic()
    emit(progress, {"event": "load_model_start", "embedding_model": config.embedding.model})
    embedder = SentenceTransformerEmbeddings(config.embedding)
    emit(progress, {"event": "load_model_finish", "seconds": time.monotonic() - load_started})

    emit(progress, {"event": "qdrant_start", "qdrant_url": config.qdrant.url, "collection": config.qdrant.collection})
    client = qdrant_client(config.qdrant.url)
    ensure_qdrant_collection(client, config.qdrant.collection, config.qdrant.vector_size)
    emit(progress, {"event": "qdrant_finish"})

    run_started = time.monotonic()
    indexed = 0
    for start in range(start_offset, end_offset, options.batch_size):
        batch_started = time.monotonic()
        batch = chunks[start : min(start + options.batch_size, end_offset)]
        embeddings = embedder.encode([embedding_text_for_chunk(chunk) for chunk in batch])
        embed_seconds = time.monotonic() - batch_started
        upsert_started = time.monotonic()
        upsert_qdrant_batch(client, config.qdrant.collection, batch, embeddings)
        upsert_seconds = time.monotonic() - upsert_started
        indexed += len(batch)
        next_offset = start + len(batch)
        elapsed_seconds = time.monotonic() - run_started
        write_index_state(
            output_artifact_dir,
            {
                "backend": "qdrant",
                "collection": config.qdrant.collection,
                "embedding_model": config.embedding.model,
                "embedding_input_version": EMBEDDING_INPUT_VERSION,
                "total_chunks": len(chunks),
                "next_offset": next_offset,
                "last_batch_size": len(batch),
                "artifact_fingerprint": artifact_fingerprint,
                "completed": next_offset >= len(chunks),
                "updated_at_unix": int(time.time()),
            },
        )
        emit(
            progress,
            {
                "event": "batch",
                "batch_start": start,
                "batch_end": next_offset,
                "batch_size": len(batch),
                "indexed_this_run": indexed,
                "chunks_to_index": end_offset - start_offset,
                "total_chunks": len(chunks),
                "next_offset": next_offset,
                "embed_seconds": embed_seconds,
                "upsert_seconds": upsert_seconds,
                "elapsed_seconds": elapsed_seconds,
                "chunks_per_second": indexed / elapsed_seconds if elapsed_seconds else 0.0,
            },
        )

    report = {
        "chunks_indexed": indexed,
        "total_chunks": len(chunks),
        "start_offset": start_offset,
        "end_offset": end_offset,
        "next_offset": end_offset,
        "complete": end_offset >= len(chunks),
        "embedding_model": config.embedding.model,
        "embedding_input_version": EMBEDDING_INPUT_VERSION,
        "backend": "qdrant",
        "qdrant_url": config.qdrant.url,
        "collection": config.qdrant.collection,
        "vector_size": config.qdrant.vector_size,
    }
    write_json(output_artifact_dir / INDEX_REPORT_FILE, report)
    emit(progress, {"event": "finish", **report})
    return report


def resolve_start_offset(artifact_dir: Path, options: IndexOptions, artifact_fingerprint: dict[str, int]) -> int:
    if not options.resume:
        return max(0, options.offset)

    state_path = artifact_dir / STATE_FILE
    if not state_path.exists():
        return max(0, options.offset)
    raw_state = read_json(state_path)
    if not isinstance(raw_state, dict):
        return max(0, options.offset)
    if raw_state.get("artifact_fingerprint") != artifact_fingerprint:
        return max(0, options.offset)
    return max(0, int(raw_state.get("next_offset") or options.offset))


def write_index_state(artifact_dir: Path, state: dict[str, object]) -> None:
    write_json(artifact_dir / STATE_FILE, state)


def file_fingerprint(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def emit(progress: Callable[[dict[str, object]], None] | None, event: dict[str, object]) -> None:
    if progress:
        progress(event)


def embedding_text_for_chunk(chunk: Chunk) -> str:
    metadata = chunk.metadata
    parts = [
        f"Title: {chunk.title}",
        f"Headings: {' > '.join(chunk.heading_path)}" if chunk.heading_path else "",
        f"Tags: {', '.join(chunk.tags)}" if chunk.tags else "",
        f"Source: {chunk.source_type}",
        f"Document: {chunk.document_id}",
        f"Chunk type: {metadata.get('chunk_type', '')}",
        f"Domain: {metadata.get('domain', '')}",
        f"Subculture: {metadata.get('subculture', '')}",
        f"Series: {chunk.series or metadata.get('series_primary', '')}",
        "",
        chunk.text,
    ]
    return "\n".join(part for part in parts if str(part).strip())


def ensure_qdrant_collection(client, collection: str, vector_size: int) -> None:
    try:
        from qdrant_client.http import models
    except ImportError as exc:
        raise RuntimeError("qdrant-client is required for cluster indexing.") from exc

    existing = {item.name for item in client.get_collections().collections}
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
    for field in ["source_type", "domain", "subculture", "chunk_type", "year", "series_primary", "page_id", "post_id"]:
        try:
            client.create_payload_index(collection_name=collection, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD)
        except Exception:
            pass


def upsert_qdrant_batch(client, collection: str, chunks, embeddings: list[list[float]]) -> None:
    from qdrant_client.http import models

    points = [
        models.PointStruct(
            id=stable_point_id(chunk.id),
            vector=embedding,
            payload=qdrant_payload(chunk),
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]
    client.upsert(collection_name=collection, points=points, wait=True)


def stable_point_id(value: str) -> str:
    import uuid

    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))
