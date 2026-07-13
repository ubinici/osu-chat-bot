from __future__ import annotations

import re

from ..domain.models import QueryIntent

TOKEN_RE = re.compile(r"[A-Za-z0-9!+#.']+")


TROUBLESHOOTING_TERMS = {
    "bug",
    "broken",
    "cant",
    "can't",
    "crash",
    "error",
    "fix",
    "freeze",
    "glitch",
    "help",
    "issue",
    "issues",
    "lag",
    "lagging",
    "problem",
    "problems",
    "slow",
    "stutter",
    "stuttering",
    "trouble",
    "troubleshoot",
    "troubleshooting",
    "wont",
    "won't",
}
PERFORMANCE_TERMS = {
    "fps",
    "frame",
    "frames",
    "framerate",
    "lag",
    "lagging",
    "latency",
    "performance",
    "slow",
    "stutter",
    "stuttering",
}
ACCESS_TERMS = {"access", "download", "get", "install", "open", "use", "using"}
DEFINITION_PATTERNS = [
    re.compile(r"\bwhat\s+(is|are)\b", re.IGNORECASE),
    re.compile(r"\bdefine\b", re.IGNORECASE),
    re.compile(r"\bmeaning\s+of\b", re.IGNORECASE),
]
OSU_QUERY_ALIASES = {
    "ar": {"approach", "rate"},
    "dt": {"double", "time", "mod"},
    "hd": {"hidden", "mod"},
    "hp": {"health", "drain"},
    "hr": {"hard", "rock", "mod"},
    "map": {"beatmap"},
    "mapper": {"beatmap", "creator"},
    "od": {"overall", "difficulty", "timing", "window"},
    "pp": {"performance", "points"},
}


def classify_query(query: str) -> QueryIntent:
    raw_tokens = {token.casefold() for token in TOKEN_RE.findall(query)}
    tokens = {normalize_token(token) for token in raw_tokens}
    labels: set[str] = set()
    expanded_terms = set(tokens)

    for token in tokens:
        expanded_terms.update(OSU_QUERY_ALIASES.get(token, set()))

    if raw_tokens.intersection(TROUBLESHOOTING_TERMS) or tokens.intersection(TROUBLESHOOTING_TERMS):
        labels.add("troubleshooting")
        expanded_terms.update({"help", "issue", "problem", "troubleshooting"})

    if raw_tokens.intersection(PERFORMANCE_TERMS) or tokens.intersection(PERFORMANCE_TERMS):
        labels.update({"troubleshooting", "performance"})
        expanded_terms.update({"lag", "performance", "frame", "fps", "latency", "stutter", "troubleshooting"})

    if raw_tokens.intersection(ACCESS_TERMS) or tokens.intersection(ACCESS_TERMS):
        labels.add("access")
        expanded_terms.update({"access", "open", "use", "download"})

    if any(pattern.search(query) for pattern in DEFINITION_PATTERNS):
        labels.add("definition")

    return QueryIntent(labels=labels, expanded_terms={term for term in expanded_terms if term})


def build_retrieval_query(query: str, intent: QueryIntent) -> str:
    """Add small, deterministic osu! hints while preserving the user's wording."""

    query_terms = {normalize_token(token) for token in TOKEN_RE.findall(query)}
    hints = sorted(term for term in intent.expanded_terms if term not in query_terms)
    if not hints:
        return query
    return f"{query}\nRelated osu! terms: {', '.join(hints)}"


def normalize_token(token: str) -> str:
    token = token.casefold().strip("'")
    if token in {"can't", "cannot"}:
        return "cant"
    if token == "won't":
        return "wont"
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ing") and len(token) > 5:
        root = token[:-3]
        if len(root) >= 2 and root[-1] == root[-2]:
            root = root[:-1]
        return root
    if token.endswith("s") and len(token) > 4 and not token.endswith("ss"):
        return token[:-1]
    return token
