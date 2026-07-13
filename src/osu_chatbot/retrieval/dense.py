from __future__ import annotations

from typing import Any

from ..config import AppConfig
from ..domain.models import Chunk, SearchResult
from ..indexing.embeddings import SentenceTransformerEmbeddings
from ..indexing.vector_store import qdrant_client


CHUNK_PAYLOAD_KEYS = {
    "chunk_id",
    "document_id",
    "source_type",
    "file_path",
    "osu_url",
    "title",
    "text",
    "chunk_index",
    "heading_path",
    "tags",
    "date",
    "series",
}


class DenseRetriever:
    """Retrieve self-contained chunks directly from a Qdrant collection."""

    def __init__(self, config: AppConfig, *, embedder: Any = None, client: Any = None):
        self.config = config
        self._embedder = embedder or SentenceTransformerEmbeddings(config.embedding)
        self._client = client or qdrant_client(config.qdrant.url)

    def search(self, query: str, limit: int) -> list[SearchResult]:
        if limit <= 0:
            return []
        query_vector = self._embedder.encode([query])[0]
        response = self._client.query_points(
            collection_name=self.config.qdrant.collection,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        results: list[SearchResult] = []
        for hit in response.points:
            payload = dict(hit.payload or {})
            try:
                chunk = chunk_from_payload(payload)
            except ValueError:
                continue
            results.append(SearchResult(chunk=chunk, score=float(hit.score)))
        return results


def chunk_from_payload(payload: dict[str, Any]) -> Chunk:
    chunk_id = str(payload.get("chunk_id") or "").strip()
    document_id = str(payload.get("document_id") or "").strip()
    text = str(payload.get("text") or "").strip()
    if not chunk_id or not document_id or not text:
        raise ValueError("Qdrant payload is missing chunk_id, document_id, or text.")

    metadata = {key: value for key, value in payload.items() if key not in CHUNK_PAYLOAD_KEYS}
    return Chunk(
        id=chunk_id,
        document_id=document_id,
        source_type=str(payload.get("source_type") or payload.get("source") or "unknown"),
        file_path=str(payload.get("file_path") or ""),
        osu_url=str(payload.get("osu_url") or ""),
        title=str(payload.get("title") or document_id),
        text=text,
        chunk_index=int(payload.get("chunk_index") or 0),
        heading_path=_string_list(payload.get("heading_path")),
        tags=_string_list(payload.get("tags")),
        date=_optional_string(payload.get("date")),
        series=_optional_string(payload.get("series")),
        metadata=metadata,
    )


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value if str(item).strip()]


def _optional_string(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None
