from __future__ import annotations

from ..config import AppConfig
from ..domain.models import Chunk
from ..indexing.embeddings import SentenceTransformerEmbeddings
from ..indexing.vector_store import qdrant_client


class DenseRetriever:
    def __init__(self, config: AppConfig, chunks: dict[str, Chunk], enabled: bool = True):
        self.config = config
        self.chunks = chunks
        self.enabled = enabled
        self._embedder = SentenceTransformerEmbeddings(config.embedding) if enabled else None
        self._collection = qdrant_client(config.qdrant.url) if enabled else None

    def scores(self, query: str) -> dict[str, float]:
        if not self.enabled or self._embedder is None or self._collection is None:
            return {}
        query_embedding = self._embedder.encode([query])[0]
        hits = self._collection.query_points(
            collection_name=self.config.qdrant.collection,
            query=query_embedding,
            limit=self.config.retrieval.dense_top_k,
            with_payload=True,
        ).points
        return {
            str(hit.payload.get("chunk_id")): float(hit.score)
            for hit in hits
            if hit.payload and hit.payload.get("chunk_id") in self.chunks
        }
