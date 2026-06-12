from osu_chatbot.corpus.chunker import build_chunks_from_record


def test_hierarchical_chunks_include_article_section_table_formula_and_citation() -> None:
    record = {
        "source": "osu-wiki",
        "page_id": "Example",
        "repo_rel_path": "Example/en.md",
        "osu_url": "https://osu.ppy.sh/wiki/en/Example",
        "title": "Example",
        "domain": "example",
        "subculture": "general",
        "lang": "en",
        "tags": ["tag"],
        "sections": [
            {
                "section_id": "intro",
                "title": "Intro",
                "heading_path": ["Example", "Intro"],
                "text": "Intro text",
                "tables": [{"table_id": "table-1", "headers": ["Name"], "rows": [["AR"]], "column_count": 1, "row_count": 1}],
                "formulae": [{"formula_id": "f1", "source": "inline_code", "expression": "OD = 10 - x"}],
            }
        ],
        "citations": [{"citation_id": "note", "text": "Citation text", "missing_definition": False}],
    }

    chunks = build_chunks_from_record(record)
    chunk_types = {chunk.metadata["chunk_type"] for chunk in chunks}

    assert {"article", "section", "table", "formula", "citation"} <= chunk_types
    assert all(chunk.osu_url == "https://osu.ppy.sh/wiki/en/Example" for chunk in chunks)
    assert any(chunk.metadata.get("heading_path") == ["Example", "Intro"] for chunk in chunks)


def test_oversized_sections_are_split_with_part_metadata() -> None:
    record = {
        "source": "osu-news",
        "post_id": "news/2026/long-post",
        "repo_rel_path": "news/2026/long-post.md",
        "osu_url": "https://osu.ppy.sh/home/news/long-post",
        "title": "Long Post",
        "sections": [
            {
                "section_id": "interview",
                "title": "Interview",
                "heading_path": ["Interview"],
                "text": "\n\n".join(["answer " * 120 for _ in range(8)]),
            }
        ],
    }

    chunks = build_chunks_from_record(record, max_chunk_chars=1200)
    section_chunks = [chunk for chunk in chunks if chunk.metadata["chunk_type"] == "section"]

    assert len(section_chunks) > 1
    assert all(len(chunk.text) < 1500 for chunk in section_chunks)
    assert {chunk.metadata.get("chunk_part_count") for chunk in section_chunks} == {len(section_chunks)}


def test_large_tables_are_split_without_omitting_rows() -> None:
    record = {
        "source": "osu-wiki",
        "page_id": "Table",
        "repo_rel_path": "Table/en.md",
        "osu_url": "https://osu.ppy.sh/wiki/en/Table",
        "title": "Table",
        "sections": [
            {
                "section_id": "rows",
                "title": "Rows",
                "heading_path": ["Rows"],
                "text": "Table section",
                "tables": [
                    {
                        "table_id": "table-1",
                        "headers": ["Name"],
                        "rows": [[f"row-{index}"] for index in range(1, 7)],
                    }
                ],
            }
        ],
    }

    chunks = build_chunks_from_record(record, max_table_rows=2, max_chunk_chars=1200)
    table_chunks = [chunk for chunk in chunks if chunk.metadata["chunk_type"] == "table"]
    text = "\n".join(chunk.text for chunk in table_chunks)

    assert len(table_chunks) == 3
    assert "row-1" in text
    assert "row-6" in text
    assert "additional rows omitted" not in text
