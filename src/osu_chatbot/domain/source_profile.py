from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import Chunk


@dataclass(frozen=True)
class SourceProfile:
    lane: str
    trust_tier: str
    weight: float


TRUST_WEIGHTS = {
    "canonical": 1.0,
    "official": 0.94,
    "community": 0.78,
    "external": 0.68,
    "unverified": 0.58,
}


def profile_for_document(document: dict[str, Any]) -> SourceProfile:
    explicit_lane = str(document.get("retrieval_lane") or "").strip()
    explicit_trust = str(document.get("trust_tier") or "").strip()
    source = str(document.get("source") or document.get("source_type") or "").casefold()
    domain = str(document.get("domain") or "").casefold()
    subculture = str(document.get("subculture") or "").casefold()

    if explicit_lane and explicit_trust:
        return SourceProfile(explicit_lane, explicit_trust, TRUST_WEIGHTS.get(explicit_trust, 0.6))

    if source == "osu-wiki":
        lane = (
            "troubleshooting"
            if "help" in {domain, subculture} or domain.startswith("help_") or domain == "performance_troubleshooting"
            else "canonical"
        )
        return SourceProfile(explicit_lane or lane, explicit_trust or "canonical", TRUST_WEIGHTS["canonical"])
    if source == "osu-news":
        return SourceProfile(explicit_lane or "temporal", explicit_trust or "official", TRUST_WEIGHTS["official"])
    if source in {"forum", "forums", "osu-forum", "community-forum"}:
        return SourceProfile(explicit_lane or "community", explicit_trust or "community", TRUST_WEIGHTS["community"])
    if source in {"chat", "discord", "irc", "message-log"}:
        return SourceProfile(explicit_lane or "community", explicit_trust or "community", TRUST_WEIGHTS["community"])
    if source in {"external", "web", "third-party"}:
        return SourceProfile(explicit_lane or "external", explicit_trust or "external", TRUST_WEIGHTS["external"])
    return SourceProfile(explicit_lane or "community", explicit_trust or "unverified", TRUST_WEIGHTS["unverified"])


def profile_for_chunk(chunk: Chunk, documents: dict[str, dict[str, Any]]) -> SourceProfile:
    lane = str(chunk.metadata.get("retrieval_lane") or "").strip()
    trust = str(chunk.metadata.get("trust_tier") or "").strip()
    if lane and trust:
        return SourceProfile(lane, trust, TRUST_WEIGHTS.get(trust, 0.6))
    document = documents.get(chunk.document_id)
    if document:
        return profile_for_document(document)
    return profile_for_document({"source": chunk.metadata.get("source") or chunk.source_type})
