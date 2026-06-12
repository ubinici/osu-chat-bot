from __future__ import annotations


class GlinerEntityExtractor:
    backend = "gliner"

    def __init__(self, *, model_name: str | None = None, threshold: float = 0.5):
        self.source_model = model_name or "urchade/gliner_medium-v2.1"
        self.threshold = threshold
        try:
            from gliner import GLiNER
        except ImportError as exc:
            raise RuntimeError(
                "GLiNER is not installed. Install the optional entity extraction dependencies with "
                '`python -m pip install -e ".[entities]"`.'
            ) from exc
        self._model = GLiNER.from_pretrained(self.source_model)

    def extract(self, text: str, labels: list[str]) -> list[dict]:
        return list(self._model.predict_entities(text, labels, threshold=self.threshold))
