from __future__ import annotations

from urllib import error, request
import json

from ..config import OllamaConfig


def generate_with_ollama(prompt: str, config: OllamaConfig) -> str:
    payload = {
        "model": config.model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": config.temperature},
    }
    req = request.Request(
        f"{config.url}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (OSError, error.URLError) as exc:
        raise RuntimeError(f"Could not reach Ollama at {config.url}. Is Ollama running with model `{config.model}`?") from exc

    return str(data.get("response", "")).strip()
