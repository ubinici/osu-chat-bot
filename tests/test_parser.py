from pathlib import Path

from osu_chatbot.corpus.parser import parse_news_file, parse_wiki_file, structure_markdown


def test_parse_wiki_file_preserves_structure(tmp_path: Path) -> None:
    root = tmp_path / "osu-wiki"
    page = root / "wiki" / "Example" / "en.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        """---
tags:
  - example tag
---

# Example page

Intro with [Target](/wiki/Target) and ::{ flag=US }::.

## Stats {id=custom-stats}

| Name | Value |
| :-- | --: |
| AR | `9.5` |

Formula: `OD = 10 - x`[^note]

![Diagram](img/diagram.png "Diagram title")

[^note]: A useful citation.
""",
        encoding="utf-8",
    )

    record = parse_wiki_file(page, root)

    assert record["schema_version"] == "2.0"
    assert record["title"] == "Example page"
    assert record["tags"] == ["example tag"]
    assert record["section_count"] == 2
    assert record["edge_count"] >= 1
    assert record["citation_count"] == 1
    assert record["table_count"] == 1
    assert record["formula_count"] >= 1
    assert record["image_count"] == 1
    assert record["sections"][1]["section_id"] == "custom-stats"


def test_parse_news_file_preserves_news_metadata(tmp_path: Path) -> None:
    root = tmp_path / "osu-wiki"
    post = root / "news" / "2026" / "2026-06-07-example-post.md"
    post.parent.mkdir(parents=True)
    post.write_text(
        """---
layout: post
title: Example News
date: 2026-06-07 20:00:00 +0000
series: community
---

![](/wiki/shared/news/example/banner.jpg)

This is the preview paragraph with [a link](https://osu.ppy.sh/home).

## Details

More news.
""",
        encoding="utf-8",
    )

    record = parse_news_file(post, root)

    assert record["source"] == "osu-news"
    assert record["post_id"] == "news/2026/2026-06-07-example-post"
    assert record["title"] == "Example News"
    assert record["date_iso"] == "2026-06-07T20:00:00+00:00"
    assert record["series_tags"] == ["community"]
    assert record["preview_paragraph"] == "This is the preview paragraph with a link."
    assert record["banner_image"] is not None


def test_extract_links_preserves_targets_with_parentheses() -> None:
    _, edges, _, _ = structure_markdown(
        "See [Double Time (lazer)](/wiki/Gameplay/Game_modifier/Double_Time_(lazer)).",
        page_id="Source",
        rel_path="Source/en.md",
    )

    assert edges[0]["link_text"] == "Double Time (lazer)"
    assert edges[0]["target_page_id"] == "Gameplay/Game_modifier/Double_Time_(lazer)"


def test_extract_links_includes_context_snippet() -> None:
    _, edges, _, _ = structure_markdown(
        "The Vocaloid Mapping Contest has [the contest's wiki page](/wiki/Contests/VMC).",
        page_id="Source",
        rel_path="Source/en.md",
    )

    assert "Vocaloid Mapping Contest" in edges[0]["context"]
    assert "contest's wiki page" in edges[0]["context"]
