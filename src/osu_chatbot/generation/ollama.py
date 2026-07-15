from __future__ import annotations

from ..config import GenerationConfig
from .base import GenerationError
from .http import post_json


class OllamaGenerator:
    def __init__(self, config: GenerationConfig):
        self.config = config

    def generate(self, prompt: str) -> str:
        headers = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        data = post_json(
            f"{self.config.url}/api/generate",
            {
                "model": self.config.model,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": {"temperature": self.config.temperature},
            },
            headers=headers,
            timeout_seconds=self.config.timeout_seconds,
            service_name=f"Ollama at {self.config.url}",
        )
        answer = str(data.get("response") or "").strip()
        if not answer:
            raise GenerationError("Ollama returned an empty answer.")
        return answer
