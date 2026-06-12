from osu_chatbot.knowledge.links import build_link_candidates, collect_link_rows


def test_link_candidates_accept_repeated_specific_internal_alias() -> None:
    docs = [
        {
            "source": "osu-wiki",
            "page_id": f"Source/{index}",
            "title": f"Source {index}",
            "repo_rel_path": f"Source/{index}/en.md",
            "link_edges": [
                {
                    "source_section_id": "intro",
                    "source_heading_path": ["Intro"],
                    "link_text": "osu!direct",
                    "link_type": "wiki_article",
                    "target_page_id": "Beatmap/osu!direct",
                }
            ],
        }
        for index in range(8)
    ]

    candidates = build_link_candidates(collect_link_rows(docs))
    candidate = next(item for item in candidates if item["alias"] == "osu!direct")

    assert candidate["decision"] == "accept"
    assert candidate["target_page_id"] == "Beatmap/osu!direct"


def test_link_candidates_reject_generic_anchors() -> None:
    docs = [
        {
            "source": "osu-wiki",
            "page_id": "Source",
            "title": "Source",
            "repo_rel_path": "Source/en.md",
            "link_edges": [
                {
                    "source_section_id": "intro",
                    "source_heading_path": ["Intro"],
                    "link_text": "click here",
                    "link_type": "wiki_article",
                    "target_page_id": "Beatmap",
                }
            ],
        }
    ]

    candidates = build_link_candidates(collect_link_rows(docs))

    assert candidates[0]["decision"] == "reject"


def test_link_candidates_reject_context_dependent_wiki_page_anchor() -> None:
    docs = [
        {
            "source": "osu-wiki",
            "page_id": "Source",
            "title": "Source",
            "repo_rel_path": "Source/en.md",
            "link_edges": [
                {
                    "source_section_id": "intro",
                    "source_heading_path": ["Intro"],
                    "link_text": "contest's wiki page",
                    "link_type": "wiki_article",
                    "target_page_id": "Contests/VMC",
                    "context": "The Vocaloid Mapping Contest has the contest's wiki page.",
                }
            ],
        }
    ]

    candidates = build_link_candidates(collect_link_rows(docs), wiki_page_ids={"Contests/VMC"})

    assert candidates[0]["decision"] == "reject"
    assert candidates[0]["examples"][0]["context"] == "The Vocaloid Mapping Contest has the contest's wiki page."


def test_link_candidates_review_ambiguous_aliases() -> None:
    docs = [
        {
            "source": "osu-wiki",
            "page_id": "Source",
            "title": "Source",
            "repo_rel_path": "Source/en.md",
            "link_edges": [
                {
                    "source_section_id": "intro",
                    "source_heading_path": ["Intro"],
                    "link_text": "ranking",
                    "link_type": "wiki_article",
                    "target_page_id": "Ranking",
                },
                {
                    "source_section_id": "intro",
                    "source_heading_path": ["Intro"],
                    "link_text": "ranking",
                    "link_type": "wiki_article",
                    "target_page_id": "Ranking_criteria",
                },
            ],
        }
    ]

    candidates = build_link_candidates(collect_link_rows(docs))

    assert candidates[0]["decision"] == "review"
    assert candidates[0]["ambiguous_target_count"] == 2


def test_link_candidates_exclude_news_targets_when_known_wiki_pages_are_provided() -> None:
    docs = [
        {
            "source": "osu-news",
            "post_id": "news/2026/example",
            "title": "Example",
            "repo_rel_path": "news/2026/example.md",
            "link_edges": [
                {
                    "source_section_id": "intro",
                    "source_heading_path": ["Intro"],
                    "link_text": "interviews",
                    "link_type": "wiki_article",
                    "target_page_id": "news/2026/other-post",
                }
            ],
        }
    ]

    assert build_link_candidates(collect_link_rows(docs), wiki_page_ids={"Beatmap"}) == []


def test_single_tournament_year_alias_is_rejected() -> None:
    docs = [
        {
            "source": "osu-wiki",
            "page_id": "Tournaments",
            "title": "Tournaments",
            "repo_rel_path": "Tournaments/en.md",
            "link_edges": [
                {
                    "source_section_id": "intro",
                    "source_heading_path": ["Intro"],
                    "link_text": "2025",
                    "link_type": "wiki_article",
                    "target_page_id": "Tournaments/OWC/2025",
                }
            ],
        }
    ]

    candidates = build_link_candidates(collect_link_rows(docs), wiki_page_ids={"Tournaments/OWC/2025"})

    assert candidates[0]["decision"] == "reject"
