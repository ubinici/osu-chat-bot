from osu_chatbot.retrieval.intent import classify_query


def test_classify_troubleshooting_performance_query() -> None:
    intent = classify_query("the game is lagging a lot, how do I fix this")

    assert "troubleshooting" in intent.labels
    assert "performance" in intent.labels
    assert {"lag", "fps", "stutter", "troubleshooting"} <= intent.expanded_terms


def test_classify_access_query() -> None:
    intent = classify_query("How to access osu!direct")

    assert "access" in intent.labels
    assert {"access", "download", "install"} <= intent.expanded_terms


def test_ambiguous_verbs_do_not_trigger_access_intent() -> None:
    assert "access" not in classify_query("how do I get better?").labels
    assert "access" not in classify_query("what can get me banned?").labels
    assert "access" not in classify_query("what menu should I open?").labels


def test_classify_definition_and_temporal_queries() -> None:
    assert "definition" in classify_query("What does AR change?").labels
    assert "temporal" in classify_query("What is the latest osu update?").labels
