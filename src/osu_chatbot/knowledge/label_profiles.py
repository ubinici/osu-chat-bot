from __future__ import annotations

from ..domain.models import Chunk

MAIN_PAGE_LABEL_PROFILE = "main-page"
OSU_ENTITY_LABEL_PROFILE = "osu-entities"

MAIN_PAGE_ENTITY_LABELS = [
    "osu account setup or access term",
    "osu client interface or file format term",
    "osu gameplay mechanic or game mode term",
    "osu beatmap editor tool or mapping term",
    "osu beatmap ranking or quality control term",
    "osu rules legal or moderation policy term",
    "osu help support or troubleshooting term",
    "osu community project contest or tournament term",
    "osu people team user group or role term",
    "osu developer api bot or integration term",
    "osu wiki maintenance or contribution term",
]

OSU_ENTITY_LABELS = [
    "osu game mode",
    "game modifier",
    "beatmap concept",
    "mapping term",
    "modding term",
    "tournament",
    "community role",
    "osu client feature",
    "ranking or scoring metric",
    "support issue",
    "rules or moderation concept",
    "osu api resource",
]

LABEL_PROFILES = {
    MAIN_PAGE_LABEL_PROFILE: MAIN_PAGE_ENTITY_LABELS,
    OSU_ENTITY_LABEL_PROFILE: OSU_ENTITY_LABELS,
}

MAIN_PAGE_PATH_LABELS = [
    (
        "osu wiki maintenance or contribution term",
        (
            "article_styling_criteria",
            "news_styling_criteria",
            "osu!_wiki/",
            "sitemap",
            "history_of_osu!",
        ),
    ),
    (
        "osu client interface or file format term",
        (
            "client/interface",
            "client/options",
            "client/keyboard_shortcuts",
            "client/program_files",
            "client/file_formats",
            "client/installation",
            "client/skin",
            "skinning",
        ),
    ),
    (
        "osu beatmap editor tool or mapping term",
        (
            "client/beatmap_editor",
            "beatmapping/mapping_techniques",
            "storyboard",
        ),
    ),
    (
        "osu beatmap ranking or quality control term",
        (
            "beatmapping/beatmap_submission",
            "beatmap_ranking_procedure",
            "ranking_criteria",
            "modding",
            "community/mappers_guild",
            "community/project_loved",
        ),
    ),
    (
        "osu gameplay mechanic or game mode term",
        (
            "game_mode",
            "gameplay/",
            "beatmap/difficulty",
            "beatmap",
            "medals",
            "performance_points",
            "ranking",
        ),
    ),
    (
        "osu rules legal or moderation policy term",
        (
            "rules",
            "legal",
        ),
    ),
    (
        "osu help support or troubleshooting term",
        (
            "help_centre",
            "performance_troubleshooting",
            "reporting_bad_behaviour",
            "people/account_support_team",
            "people/technical_support_team",
        ),
    ),
    (
        "osu community project contest or tournament term",
        (
            "community/",
            "contests",
            "tournaments",
            "beatmap_spotlights",
        ),
    ),
    (
        "osu people team user group or role term",
        (
            "people/",
            "banchobot",
        ),
    ),
    (
        "osu developer api bot or integration term",
        (
            "osu!api",
            "bot_account",
            "announcement_messages",
            "brand_identity_guidelines",
        ),
    ),
    (
        "osu account setup or access term",
        (
            "registration",
            "faq",
            "guides",
            "glossary",
        ),
    ),
]

ALWAYS_SPECIFIC_LABELS = [
    "osu gameplay mechanic or game mode term",
    "osu beatmap editor tool or mapping term",
    "osu beatmap ranking or quality control term",
    "osu community project contest or tournament term",
    "osu people team user group or role term",
    "osu developer api bot or integration term",
]


def labels_for_profile(profile: str) -> list[str]:
    try:
        return LABEL_PROFILES[profile]
    except KeyError as exc:
        valid = ", ".join(sorted(LABEL_PROFILES))
        raise ValueError(f"Unknown label profile `{profile}`. Valid profiles: {valid}") from exc


def labels_for_chunk(chunk: Chunk, labels: list[str], *, profile: str, scoped: bool) -> list[str]:
    if not scoped or profile != MAIN_PAGE_LABEL_PROFILE:
        return labels
    matched = main_page_label_for_chunk(chunk)
    if not matched:
        return labels
    scoped_labels = [matched, *ALWAYS_SPECIFIC_LABELS]
    return list(dict.fromkeys(scoped_labels))


def main_page_label_for_chunk(chunk: Chunk) -> str | None:
    haystack = f"{chunk.document_id} {chunk.file_path}".replace("\\", "/").casefold()
    for label, prefixes in MAIN_PAGE_PATH_LABELS:
        if any(prefix in haystack for prefix in prefixes):
            return label
    return None
