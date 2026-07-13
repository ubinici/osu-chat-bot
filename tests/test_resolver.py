from pathlib import Path

from osu_chatbot.domain.artifacts import write_jsonl
from osu_chatbot.domain.models import Chunk, QueryIntent
from osu_chatbot.retrieval.resolver import DocumentResolver


def document(page_id: str, title: str) -> dict:
    return {
        "source": "osu-wiki",
        "page_id": page_id,
        "title": title,
        "repo_rel_path": f"{page_id}/en.md",
        "tags": [],
        "sections": [{"title": title, "heading_path": [title]}],
    }


def test_title_acronyms_resolve_to_expected_documents(tmp_path: Path) -> None:
    docs = {
        "Beatmap/Approach_rate": document("Beatmap/Approach_rate", "Approach rate"),
        "Beatmap/Overall_difficulty": document("Beatmap/Overall_difficulty", "Overall difficulty"),
        "Performance_points": document("Performance_points", "Performance points"),
    }
    resolver = DocumentResolver(docs, tmp_path)

    assert resolver.resolve("what does AR change?", QueryIntent(), limit=3)[0].document_id == "Beatmap/Approach_rate"
    assert resolver.resolve("what does OD mean?", QueryIntent(), limit=3)[0].document_id == "Beatmap/Overall_difficulty"
    assert resolver.resolve("why did this play give more pp?", QueryIntent(), limit=3)[0].document_id == "Performance_points"


def test_chunk_text_acronyms_resolve_to_expected_documents(tmp_path: Path) -> None:
    docs = {"Beatmap/HP_drain_rate": document("Beatmap/HP_drain_rate", "HP drain rate")}
    chunks = [
        Chunk(
            id="hp::article",
            document_id="Beatmap/HP_drain_rate",
            source_type="wiki",
            file_path="Beatmap/HP_drain_rate/en.md",
            osu_url="https://osu.ppy.sh/wiki/en/Beatmap/HP_drain_rate",
            title="HP drain rate",
            text="**HP drain rate** (***HP***) controls how much health is passively lost.",
            chunk_index=0,
            heading_path=["HP drain rate"],
        )
    ]
    resolver = DocumentResolver(docs, tmp_path, chunks=chunks)

    assert resolver.resolve("what does HP do?", QueryIntent(), limit=3)[0].document_id == "Beatmap/HP_drain_rate"


def test_accepted_link_aliases_route_to_target_documents(tmp_path: Path) -> None:
    docs = {"Target/Page": document("Target/Page", "Target page")}
    write_jsonl(
        tmp_path / "link_alias_candidates.jsonl",
        [
            {
                "alias": "special alias",
                "alias_key": "special alias",
                "target_page_id": "Target/Page",
                "decision": "accept",
                "confidence": 0.95,
            }
        ],
    )
    resolver = DocumentResolver(docs, tmp_path)

    assert resolver.accepted_alias_target("special alias") == "Target/Page"
    assert resolver.resolve("what is special alias?", QueryIntent(), limit=3)[0].document_id == "Target/Page"


def test_resolver_prefers_generated_document_alias_artifact(tmp_path: Path) -> None:
    docs = {"Target/Page": document("Target/Page", "Target page")}
    write_jsonl(
        tmp_path / "document_aliases.jsonl",
        [
            {
                "alias": "artifact alias",
                "alias_key": "artifact alias",
                "document_id": "Target/Page",
                "canonical_document_id": "Target/Page",
                "source": "accepted_link",
                "confidence": 1.0,
                "retrieval_lane": "canonical",
                "trust_tier": "canonical",
            }
        ],
    )
    resolver = DocumentResolver(docs, tmp_path)

    routed = resolver.resolve("artifact alias", QueryIntent(), limit=3)

    assert routed[0].document_id == "Target/Page"
    assert routed[0].topic_id == "Target/Page"
    assert routed[0].retrieval_lane == "canonical"
    assert routed[0].trust_tier == "canonical"


def test_review_and_rejected_link_aliases_are_not_exact_routes(tmp_path: Path) -> None:
    docs = {"Target/Page": document("Target/Page", "Target page")}
    write_jsonl(
        tmp_path / "link_alias_candidates.jsonl",
        [
            {
                "alias": "needs review",
                "alias_key": "needs review",
                "target_page_id": "Target/Page",
                "decision": "review",
                "confidence": 0.9,
            },
            {
                "alias": "bad alias",
                "alias_key": "bad alias",
                "target_page_id": "Target/Page",
                "decision": "reject",
                "confidence": 1.0,
            },
        ],
    )
    resolver = DocumentResolver(docs, tmp_path)

    assert resolver.accepted_alias_target("needs review") is None
    assert resolver.accepted_alias_target("bad alias") is None
    assert resolver.resolve("needs review", QueryIntent(), limit=3) == []
    assert resolver.resolve("bad alias", QueryIntent(), limit=3) == []


def test_resolver_carries_compatible_topic_ids(tmp_path: Path) -> None:
    docs = {"Beatmap/HP_drain_rate": document("Beatmap/HP_drain_rate", "HP drain rate")}
    resolver = DocumentResolver(docs, tmp_path)

    routed = resolver.resolve("what is Beatmap Health drain?", QueryIntent(), limit=3)

    assert routed[0].document_id == "Beatmap/HP_drain_rate"
    assert routed[0].topic_id == "Beatmap/Health_drain"
    assert "Beatmap/HP_drain_rate" in routed[0].topic_document_ids
    assert "Beatmap/Health_drain" in routed[0].topic_document_ids
