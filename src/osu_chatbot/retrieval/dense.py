from __future__ import annotations

from typing import Any

from ..config import AppConfig
from ..domain.models import Chunk, SearchResult
from ..indexing.embeddings import SentenceTransformerEmbeddings
from ..indexing.vector_store import qdrant_client
from .models import RetrievalRequest


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

    def search(self, request: RetrievalRequest) -> list[SearchResult]:
        if request.limit <= 0:
            return []
        query_vector = self._embedder.encode([request.query])[0]
        query_kwargs = {
            "collection_name": self.config.qdrant.collection,
            "query": query_vector,
            "limit": request.limit,
            "with_payload": True,
        }
        query_filter = build_qdrant_filter(request)
        if query_filter is not None:
            query_kwargs["query_filter"] = query_filter
        response = self._client.query_points(
            **query_kwargs,
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

    def is_ready(self) -> bool:
        return bool(self._client.collection_exists(self.config.qdrant.collection))


def build_qdrant_filter(request: RetrievalRequest):
    if not request.source_types and not request.document_ids and not request.excluded_chunk_types:
        return None
    from qdrant_client.http import models

    must = []
    if request.source_types:
        must.append(
            models.FieldCondition(
                key="source_type",
                match=models.MatchAny(any=list(request.source_types)),
            )
        )
    if request.document_ids:
        must.append(
            models.FieldCondition(
                key="document_id",
                match=models.MatchAny(any=list(request.document_ids)),
            )
        )
    must_not = []
    if request.excluded_chunk_types:
        must_not.append(
            models.FieldCondition(
                key="chunk_type",
                match=models.MatchAny(any=list(request.excluded_chunk_types)),
            )
        )
    return models.Filter(must=must or None, must_not=must_not or None)


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
