from __future__ import annotations


def infer_subculture_from_topic(topic: str) -> str:
    value = _normalize(topic)
    if value in {"development", "osu!api"}:
        return "dev"
    if value == "tournaments":
        return "tournaments"
    if value in {"beatmap", "beatmap_information", "beatmapping", "modding"}:
        return "mapping"
    if value in {"help_centre", "performance_troubleshooting"}:
        return "help"
    if value in {"gameplay", "game_mode", "performance_points", "ranking"}:
        return "gameplay"
    if value in {"community", "people", "contests"}:
        return "community"
    if value == "client":
        return "client"
    if value in {"rules", "legal"}:
        return "moderation"
    return "general"


def infer_subculture_from_path(rel_path: str, topic: str) -> str:
    path = _normalize(rel_path)
    if "/tournaments/" in path or path.startswith("tournaments/"):
        return "tournaments"
    if "/beatmapping/" in path or path.startswith("beatmapping/"):
        return "mapping"
    if "/modding/" in path or path.startswith("modding/"):
        return "mapping"
    if "/client/" in path or path.startswith("client/"):
        return "client"
    if "/osu!api/" in path or path.startswith("osu!api/"):
        return "dev"
    if "/rules/" in path or path.startswith("rules/"):
        return "moderation"
    if "/legal/" in path or path.startswith("legal/"):
        return "moderation"
    if "/help_centre/" in path or path.startswith("help_centre/"):
        return "help"
    if "/performance_troubleshooting/" in path or path.startswith("performance_troubleshooting/"):
        return "help"
    return infer_subculture_from_topic(topic)


def infer_domain(rel_path: str) -> str:
    path = _normalize(rel_path)
    return path.split("/", 1)[0] if path else "unknown"


def _normalize(value: str) -> str:
    return (value or "").strip().lower()
