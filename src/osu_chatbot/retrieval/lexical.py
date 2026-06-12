from __future__ import annotations

import re

from .intent import normalize_token

TOKEN_RE = re.compile(r"[A-Za-z0-9!+#.]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "was",
    "were",
    "will",
    "would",
    "can",
    "could",
    "should",
    "has",
    "have",
    "had",
    "not",
    "game",
    "games",
}


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw_token in TOKEN_RE.findall(text):
        token = normalize_token(raw_token)
        if token and token not in STOPWORDS:
            tokens.append(token)
    return tokens


def section_text_parts(document: dict) -> list[str]:
    parts: list[str] = []
    sections = document.get("sections")
    if not isinstance(sections, list):
        return parts
    for section in sections:
        if not isinstance(section, dict):
            continue
        parts.append(str(section.get("title") or ""))
        heading_path = section.get("heading_path")
        if isinstance(heading_path, list):
            parts.extend(str(heading) for heading in heading_path)
    return parts


def document_id(document: dict) -> str:
    return str(document.get("page_id") or document.get("post_id") or document.get("repo_rel_path") or "")
