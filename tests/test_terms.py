from osu_chatbot.knowledge.terms import acronym_for, build_terms, detect_entities, link_alias_matches_target


def test_acronym_for_multiword_terms() -> None:
    assert acronym_for("Nomination Assessment Team") == "NAT"


def test_build_terms_from_titles_tags_headings_and_links() -> None:
    docs = [
        {
            "source": "osu-wiki",
            "page_id": "Beatmap",
            "repo_rel_path": "Beatmap/en.md",
            "title": "Beatmap",
            "tags": ["mapset"],
            "sections": [{"title": "Difficulty", "heading_path": ["Beatmap", "Difficulty"]}],
            "link_edges": [{"link_text": "osu!direct", "target_page_id": "osu!direct"}],
        }
    ]

    entities = build_terms(docs)
    detected = detect_entities("How does osu!direct work for beatmaps?", entities)
    canonical = {entity.canonical for entity in detected}

    assert "Beatmap" in {entity.canonical for entity in entities}
    assert "osu!direct" in canonical
    assert "Beatmap" in canonical


def test_link_alias_must_resemble_target() -> None:
    assert link_alias_matches_target("osu!direct", "osu!direct")
    assert not link_alias_matches_target("Beatmap", "Anchor")


def test_parent_path_parts_do_not_become_child_aliases() -> None:
    docs = [
        {
            "source": "osu-wiki",
            "page_id": "Beatmap/Pattern/osu!mania/Anchor",
            "repo_rel_path": "Beatmap/Pattern/osu!mania/Anchor/en.md",
            "title": "Anchor",
            "sections": [],
            "link_edges": [],
        }
    ]

    detected = detect_entities("What is a beatmap?", build_terms(docs))

    assert "Anchor" not in {entity.canonical for entity in detected}


def test_stopword_acronyms_do_not_trigger_entities() -> None:
    docs = [
        {
            "source": "osu-news",
            "post_id": "news/2017/example",
            "repo_rel_path": "news/2017/example.md",
            "title": "Beatmap Spotlights",
            "sections": [{"title": "What are Spotlights?", "heading_path": ["What are Spotlights?"]}],
            "link_edges": [],
        }
    ]

    entities = build_terms(docs)
    spotlights = next(entity for entity in entities if entity.canonical == "What are Spotlights?")

    assert "WAS" not in spotlights.aliases
    assert "What are Spotlights?" not in {entity.canonical for entity in detect_entities("when was bosphorus invitational", entities)}


def test_terms_merge_case_variants_and_skip_year_only_entities() -> None:
    docs = [
        {
            "source": "osu-news",
            "post_id": "news/2026/example",
            "repo_rel_path": "news/2026/example.md",
            "title": "BanchoBot",
            "year": "2026",
            "sections": [{"title": "banchobot", "heading_path": ["banchobot"]}],
            "link_edges": [],
        }
    ]

    entities = build_terms(docs)
    canonicals = [entity.canonical for entity in entities]
    banchobot_entities = [entity for entity in entities if entity.canonical.casefold() == "banchobot"]

    assert len(banchobot_entities) == 1
    assert "2026" not in canonicals


def test_osu_prefix_terms_prefer_specific_entity_over_generic_osu() -> None:
    entities = build_terms([])

    detected = detect_entities("How to access osu!direct?", entities)

    assert [entity.canonical for entity in detected] == ["osu!direct"]


def test_osu_prefix_terms_support_common_spacing_variants() -> None:
    entities = build_terms([])

    assert [entity.canonical for entity in detect_entities("How to access osu direct?", entities)] == ["osu!direct"]
    assert [entity.canonical for entity in detect_entities("How to get osu supporter?", entities)] == ["osu!supporter"]


def test_common_osu_compound_terms_are_detected() -> None:
    entities = build_terms([])
    detected = {entity.canonical for entity in detect_entities("Who won the osu! world cup?", entities)}

    assert "osu! World Cup" in detected
    assert "osu!" not in detected
