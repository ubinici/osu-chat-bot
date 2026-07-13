from osu_chatbot.domain.models import Chunk
from osu_chatbot.domain.source_profile import profile_for_chunk, profile_for_document


def test_source_profile_classifies_wiki_news_and_messy_sources() -> None:
    assert profile_for_document({"source": "osu-wiki", "domain": "beatmap"}).lane == "canonical"
    assert profile_for_document({"source": "osu-wiki", "domain": "help_centre"}).lane == "troubleshooting"
    assert profile_for_document({"source": "osu-news"}).trust_tier == "official"
    assert profile_for_document({"source": "forum"}).trust_tier == "community"
    assert profile_for_document({"source": "external"}).lane == "external"


def test_chunk_source_profile_uses_metadata_override() -> None:
    chunk = Chunk(
        id="chat::1",
        document_id="chat-thread",
        source_type="chat",
        file_path="chat/thread.jsonl",
        osu_url="",
        title="Chat thread",
        text="Players report a workaround.",
        chunk_index=0,
        metadata={"retrieval_lane": "troubleshooting", "trust_tier": "community"},
    )

    profile = profile_for_chunk(chunk, {})

    assert profile.lane == "troubleshooting"
    assert profile.trust_tier == "community"
