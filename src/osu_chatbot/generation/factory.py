from __future__ import annotations

from ..config import GenerationConfig
from .base import TextGenerator
from .ollama import OllamaGenerator
from .openai_compatible import OpenAICompatibleGenerator


def create_generator(config: GenerationConfig) -> TextGenerator:
    if config.provider == "ollama":
        return OllamaGenerator(config)
    if config.provider == "openai-compatible":
        return OpenAICompatibleGenerator(config)
    raise ValueError(
        f"Unsupported generation provider `{config.provider}`. "
        "Choose `ollama` or `openai-compatible`."
    )
