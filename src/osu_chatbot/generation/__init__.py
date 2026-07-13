"""Provider-neutral answer generation clients and prompts."""

from .base import GenerationError, TextGenerator
from .factory import create_generator

__all__ = ["GenerationError", "TextGenerator", "create_generator"]
