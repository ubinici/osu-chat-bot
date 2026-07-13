from osu_chatbot.knowledge.topics import build_canonical_topic_rows, topic_rows_by_document


def document(page_id: str, title: str, *, source: str = "osu-wiki", topic_id: str | None = None) -> dict:
    doc = {
        "source": source,
        "page_id": page_id,
        "title": title,
        "repo_rel_path": f"{page_id}/en.md",
        "tags": [],
        "sections": [{"title": title, "heading_path": [title]}],
    }
    if topic_id:
        doc["topic_id"] = topic_id
    return doc


def test_topics_bridge_compatible_document_ids() -> None:
    rows = build_canonical_topic_rows(
        [
            document("Beatmap/HP_drain_rate", "HP drain rate"),
            document("Client/Interface/Chat_console", "Chat console"),
        ]
    )
    by_document = topic_rows_by_document(rows)

    hp_topic = by_document["Beatmap/HP_drain_rate"]
    chat_topic = by_document["Client/Interface/Chat_console"]

    assert hp_topic["topic_id"] == "Beatmap/Health_drain"
    assert "Beatmap/Health_drain" in hp_topic["equivalent_document_ids"]
    assert "Beatmap/HP_drain_rate" in hp_topic["equivalent_document_ids"]
    assert chat_topic["topic_id"] == "Chat_console"
    assert "Client/Interface/Chat_console" in chat_topic["document_ids"]


def test_topics_attach_community_documents_to_canonical_topic() -> None:
    rows = build_canonical_topic_rows(
        [
            document("Beatmap/Approach_rate", "Approach rate"),
            document("forum/ar-thread", "AR thread", source="forum", topic_id="Beatmap/Approach_rate"),
        ]
    )
    by_document = topic_rows_by_document(rows)
    topic = by_document["forum/ar-thread"]

    assert topic["topic_id"] == "Beatmap/Approach_rate"
    assert topic["canonical_document_id"] == "Beatmap/Approach_rate"
    assert set(topic["document_ids"]) == {"Beatmap/Approach_rate", "forum/ar-thread"}
    assert topic["retrieval_lane"] == "canonical"
