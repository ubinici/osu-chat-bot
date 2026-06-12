from pathlib import Path

from osu_chatbot.corpus.markdown import parse_frontmatter, wiki_url_for_file
from osu_chatbot.corpus.parser import extract_links, infer_title_from_body


def test_parse_frontmatter_with_tags() -> None:
    frontmatter, body = parse_frontmatter("---\ntags:\n  - mapset\n  - beatmapset\n---\n\n# Beatmap\nBody")

    assert frontmatter == {"tags": ["mapset", "beatmapset"]}
    assert body.startswith("# Beatmap")


def test_extract_title_uses_first_h1() -> None:
    assert infer_title_from_body("## Not it\n# Beatmap\nText", "Fallback") == "Beatmap"


def test_extract_links_keeps_osu_wiki_links() -> None:
    links = extract_links(
        markdown="[Difficulty](/wiki/Beatmap/Difficulty) [Other](https://example.com)",
        current_page_id="Beatmap",
        current_rel_path="Beatmap/en.md",
        source_section_id="intro",
        source_heading_path=["Beatmap"],
        link_defs={},
    )

    assert links[0]["link_text"] == "Difficulty"
    assert links[0]["target_page_id"] == "Beatmap/Difficulty"


def test_wiki_and_news_urls() -> None:
    root = Path("database/osu-wiki")

    assert wiki_url_for_file(root, Path("database/osu-wiki/wiki/Beatmap/en.md")) == "https://osu.ppy.sh/wiki/en/Beatmap"
    assert (
        wiki_url_for_file(root, Path("database/osu-wiki/news/2026/2026-06-07-example.md"))
        == "https://osu.ppy.sh/home/news/2026-06-07-example"
    )
