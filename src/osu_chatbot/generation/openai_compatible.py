from __future__ import annotations

from typing import Any

from ..config import GenerationConfig
from .base import GenerationError
from .http import post_json


class OpenAICompatibleGenerator:
    """Generate through an OpenAI-compatible chat-completions endpoint."""

    def __init__(self, config: GenerationConfig):
        self.config = config

    def generate(self, prompt: str) -> str:
        headers = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        data = post_json(
            f"{self.config.url}/chat/completions",
            {
                "model": self.config.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.config.temperature,
                "stream": False,
            },
            headers=headers,
            timeout_seconds=self.config.timeout_seconds,
            service_name=f"generation endpoint at {self.config.url}",
        )
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise GenerationError("The generation endpoint returned an unexpected response.") from exc

        answer = _content_text(content)
        if not answer:
            raise GenerationError("The generation endpoint returned an empty answer.")
        return answer


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(
            str(part.get("text") or "").strip()
            for part in content
            if isinstance(part, dict) and part.get("text")
        ).strip()
    return ""
