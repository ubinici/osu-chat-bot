from __future__ import annotations

from ..config import EmbeddingConfig


class SentenceTransformerEmbeddings:
    def __init__(self, config: str | EmbeddingConfig):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for indexing and dense retrieval. "
                "Install dependencies with: python -m pip install -e ."
            ) from exc

        if isinstance(config, str):
            model_name = config
            kwargs = {}
        else:
            model_name = config.model
            kwargs = {
                key: value
                for key, value in {
                    "cache_folder": config.cache_folder,
                    "device": config.device,
                    "local_files_only": config.local_files_only,
                }.items()
                if value not in {None, False}
            }

        self.model_name = model_name
        try:
            self.model = SentenceTransformer(model_name, **kwargs)
        except TypeError:
            kwargs.pop("local_files_only", None)
            self.model = SentenceTransformer(model_name, **kwargs)

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [vector.tolist() for vector in vectors]
