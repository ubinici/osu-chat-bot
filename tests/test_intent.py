from osu_chatbot.retrieval.intent import classify_query


def test_classify_troubleshooting_performance_query() -> None:
    intent = classify_query("the game is lagging a lot, how do I fix this")

    assert "troubleshooting" in intent.labels
    assert "performance" in intent.labels
    assert "lag" in intent.expanded_terms
    assert intent.document_hints["Performance_troubleshooting"] > intent.document_hints["Help_centre/Client"]


def test_classify_access_query() -> None:
    intent = classify_query("How to access osu!direct")

    assert "access" in intent.labels
    assert {"access", "open", "download"} <= intent.expanded_terms
