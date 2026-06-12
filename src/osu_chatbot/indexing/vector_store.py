from __future__ import annotations

from ..domain.models import Chunk


def qdrant_client(url: str):
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise RuntimeError("qdrant-client is required for Qdrant indexing.") from exc

    if url.startswith("file://"):
        return QdrantClient(path=url[len("file://") :])
    return QdrantClient(url=url)


def qdrant_payload(chunk: Chunk) -> dict:
    payload = {
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "source_type": chunk.source_type,
        "file_path": chunk.file_path,
        "osu_url": chunk.osu_url,
        "title": chunk.title,
        "text": chunk.text,
        "chunk_index": chunk.chunk_index,
        "heading_path": chunk.heading_path,
        "tags": chunk.tags,
        "date": chunk.date,
        "series": chunk.series,
        **chunk.metadata,
    }
    return {key: value for key, value in payload.items() if value is not None}
