from pathlib import Path

from osu_chatbot.domain.artifacts import (
    DOCUMENT_ALIASES_FILE,
    DOCUMENTS_FILE,
    LINK_ALIAS_CANDIDATES_FILE,
    read_jsonl,
    write_jsonl,
)
from osu_chatbot.knowledge.aliases import build_document_aliases


def test_build_document_aliases_supports_independent_and_related_sources(tmp_path: Path) -> None:
    write_jsonl(
        tmp_path / DOCUMENTS_FILE,
        [
            {
                "source": "osu-wiki",
                "page_id": "Beatmap/Approach_rate",
                "title": "Approach rate",
            },
            {
                "source": "community-discord",
                "source_type": "community",
                "id": "discord:123",
                "title": "Mapper explanation",
                "aliases": ["community AR explanation"],
                "relations": [
                    {"type": "mentions", "document_id": "Beatmap/Approach_rate"},
                    {"type": "parent", "document_id": "Beatmapping"},
                ],
            },
        ],
    )
    write_jsonl(
        tmp_path / LINK_ALIAS_CANDIDATES_FILE,
        [
            {
                "alias": "AR setting",
                "target_page_id": "Beatmap/Approach_rate",
                "decision": "accept",
                "confidence": 0.98,
            },
            {
                "alias": "ambiguous AR",
                "target_page_id": "Beatmap/Approach_rate",
                "decision": "review",
                "confidence": 0.8,
            },
        ],
    )

    report = build_document_aliases(tmp_path)
    rows = read_jsonl(tmp_path / DOCUMENT_ALIASES_FILE)

    assert report == {"documents": 2, "accepted_link_aliases": 1, "aliases": 5}
    ar_setting = next(
        row
        for row in rows
        if row["alias_key"] == "ar setting" and row["source"] == "accepted_link"
    )
    assert ar_setting["preference_strength"] == "soft"
    approach_rate = next(
        row
        for row in rows
        if row["alias_key"] == "approach rate" and row["source"] == "title"
    )
    assert approach_rate["preference_strength"] == "strong"
    community = next(row for row in rows if row["alias_key"] == "community ar explanation")
    assert community["canonical_document_id"] == "discord:123"
    assert community["topic_document_ids"] == ["discord:123", "Beatmapping"]
    assert community["target_source"] == "community-discord"
    assert community["preference_strength"] == "soft"
    assert not any(row["alias_key"] == "ambiguous ar" for row in rows)
