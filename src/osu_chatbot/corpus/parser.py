from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
import posixpath
import re

from .markdown import parse_frontmatter, wiki_url_for_file
from .taxonomy import infer_domain, infer_subculture_from_path

RE_HTML_COMMENT = re.compile(r"<!--.*?-->", flags=re.DOTALL)
RE_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", flags=re.DOTALL)
RE_DIRECTIVE = re.compile(r"::\{\s*([^}=]+)\s*=\s*([^}]+)\s*\}::")
RE_CONTAINER_MARKER_LINE = re.compile(r"^\s*:::\s*([A-Za-z0-9_-]+)?\s*$", flags=re.MULTILINE)
RE_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
RE_HEADING_CUSTOM_ID = re.compile(r"^(.*?)\s*\{id=([^}]+)\}\s*$")
RE_FOOTNOTE_DEF = re.compile(r"^\[\^([^\]\s\\][^\]]*)\]:\s*(.*)$")
RE_FOOTNOTE_REF = re.compile(r"(?<!\\)\[\^([^\]\s\\][^\]]*)\]")
RE_LINK_DEF = re.compile(r"^\[(?!\^)([^\]]+)\]:\s*(\S+)(?:\s+\"([^\"]+)\")?\s*$")
RE_INLINE_LINK = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)")
RE_REFERENCE_LINK = re.compile(r"(?<!!)\[([^\]]+)\]\[([^\]]+)\]")
RE_BARE_URL = re.compile(r"https?://[^\s<>\"]+")
RE_INLINE_CODE = re.compile(r"`([^`\n]+)`")
RE_FENCED_BLOCK = re.compile(r"```(?:[^\n]*)\n(.*?)```", flags=re.DOTALL)
RE_FENCED_BLOCK_ANY = re.compile(r"```(?:[^\n]*)\n.*?```", flags=re.DOTALL)
RE_MD_IMAGE_INLINE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"([^\"]*)\")?\)")
RE_MD_IMAGE_REF = re.compile(r"!\[([^\]]*)\]\[([^\]]+)\]")
RE_ID_MARKER_LINE = re.compile(r"^\{id=([^}]+)\}\s*$", flags=re.MULTILINE)
RE_NEWS_FILENAME = re.compile(r"^(\d{4})-(\d{2})-(\d{2})-(.+)$")
RE_TITLE_MARKDOWN = re.compile(r"[*_`~\[\]()]")

ASSET_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp"}
PPY_HOSTS = {"osu.ppy.sh", "www.osu.ppy.sh", "blog.ppy.sh", "assets.ppy.sh", "packs.ppy.sh"}


def parse_wiki_file(file_path: Path, osu_wiki_root: Path, lang: str = "en") -> dict[str, Any]:
    wiki_root = osu_wiki_root / "wiki"
    rel_path = file_path.relative_to(wiki_root).as_posix()
    page_id = rel_path.removesuffix(f"/{lang}.md").removesuffix(f"{lang}.md")
    topic = rel_path.split("/", 1)[0] if rel_path else page_id
    raw = file_path.read_text(encoding="utf-8")
    frontmatter, body_raw = parse_frontmatter(raw)
    body = RE_HTML_COMMENT.sub("", body_raw)
    title = infer_title_from_body(body, fallback=page_id)
    tags = normalize_tags(frontmatter.get("tags", []))
    sections, link_edges, citations, stats = structure_markdown(body, page_id=page_id, rel_path=rel_path)

    return {
        "schema_version": "2.0",
        "source": "osu-wiki",
        "page_id": page_id,
        "repo_rel_path": rel_path,
        "osu_url": wiki_url_for_file(osu_wiki_root, file_path),
        "topic": topic,
        "domain": infer_domain(rel_path),
        "subculture": infer_subculture_from_path(rel_path, topic),
        "lang": lang,
        "title": title,
        "tags": tags,
        "is_stub": bool(frontmatter.get("stub", False)),
        "section_count": len(sections),
        "edge_count": len(link_edges),
        "citation_count": len(citations),
        "directive_count": stats["directives_total"],
        "table_count": stats["tables_total"],
        "formula_count": stats["formulas_total"],
        "image_count": stats["images_total"],
        "sections": sections,
        "link_edges": link_edges,
        "citations": citations,
        "stats": stats,
    }


def parse_news_file(file_path: Path, osu_wiki_root: Path) -> dict[str, Any]:
    rel_path = file_path.relative_to(osu_wiki_root).as_posix()
    raw = file_path.read_text(encoding="utf-8")
    frontmatter, body_raw = parse_frontmatter(raw)
    body = RE_HTML_COMMENT.sub("", body_raw)
    stem = file_path.stem
    match = RE_NEWS_FILENAME.match(stem)
    filename_date = f"{match.group(1)}-{match.group(2)}-{match.group(3)}" if match else None
    slug = match.group(4) if match else stem
    year = rel_path.split("/")[1] if "/" in rel_path else ""
    post_id = rel_path.removesuffix(".md")
    title = clean_scalar(frontmatter.get("title")) or stem.replace("-", " ")
    layout = clean_scalar(frontmatter.get("layout")).lower()
    date_raw = clean_scalar(frontmatter.get("date"))
    series_tags = normalize_series(frontmatter.get("series"))
    sections, link_edges, citations, stats = structure_markdown(body, page_id=post_id, rel_path=rel_path)
    preview = extract_preview_paragraph(body)
    banner_image = extract_banner_image_from_sections(sections)
    style = {
        "layout_is_post": layout == "post",
        "has_series": bool(series_tags),
        "date_parsed": parse_news_date(date_raw) is not None,
        "has_preview_paragraph": bool(preview),
        "has_banner_image": banner_image is not None,
        "banner_alt_is_empty": bool(banner_image and not str(banner_image.get("alt", "")).strip()),
        "no_h1_in_body": RE_HEADING.search(RE_FENCED_BLOCK_ANY.sub("", body)) is None,
        "no_footnotes_in_body": (
            RE_FOOTNOTE_REF.search(RE_FENCED_BLOCK_ANY.sub("", body)) is None
            and RE_FOOTNOTE_DEF.search(RE_FENCED_BLOCK_ANY.sub("", body)) is None
        ),
        "title_has_no_markdown": RE_TITLE_MARKDOWN.search(title) is None,
    }
    style_issues = derive_style_issues(style)
    stats = {
        **stats,
        "missing_series": 0 if series_tags else 1,
        "invalid_layout": 0 if style["layout_is_post"] else 1,
        "missing_banner_image": 0 if style["has_banner_image"] else 1,
        "has_h1_heading": 0 if style["no_h1_in_body"] else 1,
        "has_footnotes": 0 if style["no_footnotes_in_body"] else 1,
        "banner_alt_nonempty": 0 if style["banner_alt_is_empty"] or not banner_image else 1,
    }

    return {
        "schema_version": "1.0",
        "source": "osu-news",
        "post_id": post_id,
        "repo_rel_path": rel_path,
        "osu_url": wiki_url_for_file(osu_wiki_root, file_path),
        "domain": "news",
        "subculture": series_tags[0] if series_tags else "news",
        "year": year,
        "slug": slug,
        "lang": "en",
        "title": title,
        "layout": layout,
        "date_raw": date_raw,
        "date_iso": parse_news_date(date_raw),
        "filename_date": filename_date,
        "series_tags": series_tags,
        "series_primary": series_tags[0] if series_tags else None,
        "has_series": bool(series_tags),
        "preview_paragraph": preview,
        "preview_char_count": len(preview),
        "preview_word_count": len(preview.split()),
        "banner_image": banner_image,
        "style": style,
        "style_issues": style_issues,
        "section_count": len(sections),
        "edge_count": len(link_edges),
        "citation_count": len(citations),
        "table_count": stats["tables_total"],
        "formula_count": stats["formulas_total"],
        "image_count": stats["images_total"],
        "sections": sections,
        "link_edges": link_edges,
        "citations": citations,
        "stats": stats,
    }


def structure_markdown(body: str, *, page_id: str, rel_path: str) -> tuple[list[dict], list[dict], list[dict], dict]:
    link_defs = extract_link_definitions(body.splitlines())
    footnotes = extract_footnote_definitions(body.splitlines())
    raw_sections = build_sections(body, page_id)
    section_records: list[dict[str, Any]] = []
    edge_records: list[dict[str, Any]] = []
    citation_usage: dict[str, set[str]] = {key: set() for key in footnotes}
    edge_type_counts: dict[str, int] = {}
    directive_key_counts: dict[str, int] = {}
    totals = {"directives": 0, "tables": 0, "formulas": 0, "images": 0}

    for section in raw_sections:
        section_id = section["section_id"]
        section_md = strip_definition_lines(section["markdown"])
        inline_anchor_ids = extract_inline_anchor_ids(section_md)
        section_md = strip_inline_anchor_ids(section_md)
        tables, section_md_no_tables = extract_tables_and_strip(section_md)
        section_md_main = strip_container_markers(section_md_no_tables)
        semantic_md = strip_code_for_semantic_extraction(section_md_main)
        formulae = extract_formulas(section_md)
        images = extract_images(
            markdown=semantic_md,
            current_page_id=page_id,
            current_rel_path=rel_path,
            link_defs=link_defs,
        )
        directives = extract_directives(semantic_md) + extract_container_directives(section_md_no_tables)
        for directive in directives:
            directive_key_counts[directive["key"]] = directive_key_counts.get(directive["key"], 0) + 1
        totals["directives"] += len(directives)
        totals["tables"] += len(tables)
        totals["formulas"] += len(formulae)
        totals["images"] += len(images)

        edges = extract_links(
            markdown=semantic_md,
            current_page_id=page_id,
            current_rel_path=rel_path,
            source_section_id=section_id,
            source_heading_path=section["heading_path"],
            link_defs=link_defs,
        )
        edge_start = len(edges)
        for table in tables:
            table_edges = extract_links(
                markdown=strip_code_for_semantic_extraction(table["raw_markdown"]),
                current_page_id=page_id,
                current_rel_path=rel_path,
                source_section_id=section_id,
                source_heading_path=section["heading_path"],
                link_defs=link_defs,
                edge_start=edge_start,
            )
            edges.extend(table_edges)
            edge_start += len(table_edges)
        for edge in edges:
            link_type = edge["link_type"]
            edge_type_counts[link_type] = edge_type_counts.get(link_type, 0) + 1
        edge_records.extend(edges)

        refs = set(RE_FOOTNOTE_REF.findall(semantic_md))
        for table in tables:
            refs.update(RE_FOOTNOTE_REF.findall(strip_code_for_semantic_extraction(table["raw_markdown"])))
        for ref in refs:
            citation_usage.setdefault(ref, set()).add(section_id)

        text = markdown_to_text(section_md_main)
        table_text = tables_to_text(tables)
        if table_text:
            text = f"{text}\n\n{table_text}".strip()
        formula_text = formulas_to_text(formulae)
        if formula_text:
            text = f"{text}\n\n{formula_text}".strip()

        anchor_ids = list(dict.fromkeys([section_id, *inline_anchor_ids]))
        section_records.append(
            {
                "section_id": section_id,
                "parent_section_id": section["parent_section_id"],
                "level": section["level"],
                "title": section["title"],
                "heading_path": section["heading_path"],
                "anchor_ids": anchor_ids,
                "start_line": section["start_line"],
                "end_line": section["end_line"],
                "directives": directives,
                "tables": tables,
                "table_count": len(tables),
                "formulae": formulae,
                "formula_count": len(formulae),
                "images": images,
                "image_count": len(images),
                "citation_refs": sorted(refs),
                "text": text,
                "char_count": len(text),
                "word_count": len(text.split()),
            }
        )

    citation_records = []
    for key, definition in footnotes.items():
        citation_records.append(
            {
                "citation_id": key,
                "text": markdown_to_text(definition["text"]),
                "raw_text": definition["text"],
                "defined_at_line": definition["line"],
                "missing_definition": False,
                "referenced_in_sections": sorted(citation_usage.get(key, set())),
                "links": extract_links(
                    markdown=strip_code_for_semantic_extraction(definition["text"]),
                    current_page_id=page_id,
                    current_rel_path=rel_path,
                    source_section_id=f"citation:{key}",
                    source_heading_path=[],
                    link_defs=link_defs,
                ),
            }
        )
    for key in sorted(citation_usage):
        if key not in footnotes:
            citation_records.append(
                {
                    "citation_id": key,
                    "text": "",
                    "raw_text": "",
                    "defined_at_line": None,
                    "missing_definition": True,
                    "referenced_in_sections": sorted(citation_usage.get(key, set())),
                    "links": [],
                }
            )

    stats = {
        "sections_total": len(section_records),
        "links_total": len(edge_records),
        "citations_total": len(citation_records),
        "citations_missing_definitions": sum(1 for item in citation_records if item["missing_definition"]),
        "directives_total": totals["directives"],
        "tables_total": totals["tables"],
        "formulas_total": totals["formulas"],
        "images_total": totals["images"],
        "link_type_wiki_article": edge_type_counts.get("wiki_article", 0),
        "link_type_wiki_section": edge_type_counts.get("wiki_section", 0),
        "link_type_wiki_asset": edge_type_counts.get("wiki_asset", 0),
        "link_type_ppy_internal": edge_type_counts.get("ppy_internal", 0),
        "link_type_external": edge_type_counts.get("external", 0),
        "link_type_mailto": edge_type_counts.get("mailto", 0),
        "link_type_other": edge_type_counts.get("other", 0),
    }
    for key, count in sorted(directive_key_counts.items()):
        stats[f"directive_key_{key}"] = count
    return section_records, edge_records, citation_records, stats


def build_sections(markdown: str, page_id: str) -> list[dict[str, Any]]:
    lines = markdown.splitlines()
    sections: list[dict[str, Any]] = []
    stack: list[dict[str, Any]] = []
    used_ids: dict[str, int] = {}
    current: dict[str, Any] | None = None

    def finalize(section: dict[str, Any], end_line: int) -> None:
        section["end_line"] = max(section["start_line"], end_line)
        section["markdown"] = "\n".join(section.pop("_content_lines")).strip()
        sections.append(section)

    for idx, raw_line in enumerate(lines, start=1):
        match = RE_HEADING.match(raw_line)
        if not match:
            if current is None:
                current = {
                    "section_id": "__preamble",
                    "parent_section_id": None,
                    "level": 0,
                    "title": "Preamble",
                    "heading_path": [],
                    "start_line": 1,
                    "_content_lines": [],
                }
            current["_content_lines"].append(raw_line)
            continue

        level = len(match.group(1))
        title, custom_id = strip_heading_custom_id(match.group(2).strip())
        if current is not None:
            finalize(current, idx - 1)
        while stack and stack[-1]["level"] >= level:
            stack.pop()
        parent_id = stack[-1]["section_id"] if stack else None
        base_id = custom_id or slugify(title)
        count = used_ids.get(base_id, 0)
        section_id = base_id if count == 0 else f"{base_id}-{count + 1}"
        used_ids[base_id] = count + 1
        heading_path = [node["title"] for node in stack] + [title]
        current = {
            "section_id": section_id,
            "parent_section_id": parent_id,
            "level": level,
            "title": title,
            "heading_path": heading_path,
            "start_line": idx,
            "_content_lines": [],
        }
        stack.append({"section_id": section_id, "level": level, "title": title})

    if current is not None:
        finalize(current, len(lines))
    if not sections:
        sections.append(
            {
                "section_id": slugify(page_id) or "root",
                "parent_section_id": None,
                "level": 1,
                "title": page_id,
                "heading_path": [page_id],
                "start_line": 1,
                "end_line": max(1, len(lines)),
                "markdown": markdown.strip(),
            }
        )
    return sections


def extract_link_definitions(lines: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    in_fence = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = RE_LINK_DEF.match(stripped)
        if match:
            out[match.group(1).strip().lower()] = clean_link_target(match.group(2))
    return out


def extract_footnote_definitions(lines: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    index = 0
    in_fence = False
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            index += 1
            continue
        if in_fence:
            index += 1
            continue
        match = RE_FOOTNOTE_DEF.match(lines[index].rstrip())
        if not match:
            index += 1
            continue
        key = match.group(1).strip()
        text_lines = [match.group(2).strip()]
        next_index = index + 1
        while next_index < len(lines) and (lines[next_index].startswith("  ") or lines[next_index].startswith("\t")):
            text_lines.append(lines[next_index].strip())
            next_index += 1
        out[key] = {"line": index + 1, "text": "\n".join(line for line in text_lines if line)}
        index = next_index
    return out


def extract_tables_and_strip(markdown: str) -> tuple[list[dict[str, Any]], str]:
    lines = markdown.splitlines()
    tables: list[dict[str, Any]] = []
    kept: list[str] = []
    index = 0
    in_fence = False
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            kept.append(line)
            index += 1
            continue
        if in_fence or index + 1 >= len(lines) or not is_table_header_and_separator(lines[index], lines[index + 1]):
            kept.append(line)
            index += 1
            continue

        raw_lines = [lines[index], lines[index + 1]]
        row_lines: list[str] = []
        next_index = index + 2
        while next_index < len(lines) and looks_like_table_row(lines[next_index]):
            row_lines.append(lines[next_index])
            raw_lines.append(lines[next_index])
            next_index += 1
        if not row_lines:
            kept.append(line)
            index += 1
            continue

        headers = [markdown_to_text(cell) for cell in split_table_cells(lines[index])]
        separators = split_table_cells(lines[index + 1])
        rows = [[markdown_to_text(cell) for cell in split_table_cells(row)] for row in row_lines]
        width = max(len(headers), len(separators), *(len(row) for row in rows))
        headers = normalize_row(headers, width)
        rows = [normalize_row(row, width) for row in rows]
        for idx, header in enumerate(headers):
            if not header.strip():
                headers[idx] = f"col_{idx + 1}"
        tables.append(
            {
                "table_id": f"table-{len(tables) + 1}",
                "start_line_in_section": index + 1,
                "end_line_in_section": next_index,
                "column_count": width,
                "row_count": len(rows),
                "headers": headers,
                "alignments": parse_table_alignments(normalize_row(separators, width)),
                "rows": rows,
                "raw_markdown": "\n".join(raw_lines),
            }
        )
        index = next_index
    return tables, "\n".join(kept).strip()


def extract_links(
    *,
    markdown: str,
    current_page_id: str,
    current_rel_path: str,
    source_section_id: str,
    source_heading_path: list[str],
    link_defs: dict[str, str],
    edge_start: int = 0,
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    edge_index = edge_start
    occupied_ranges: list[tuple[int, int]] = []
    for match in iter_inline_link_matches(markdown):
        edge_index += 1
        occupied_ranges.append((match["start"], match["end"]))
        edges.append(
            build_edge(
                edge_index,
                source_section_id,
                source_heading_path,
                match["text"],
                clean_link_target(match["target"]),
                current_page_id,
                current_rel_path,
                False,
                None,
                link_context(markdown, match["start"], match["end"]),
            )
        )
    for match in RE_REFERENCE_LINK.finditer(markdown):
        edge_index += 1
        ref_key = match.group(2).strip()
        target = link_defs.get(ref_key.lower(), ref_key)
        edges.append(
            build_edge(
                edge_index,
                source_section_id,
                source_heading_path,
                match.group(1),
                target,
                current_page_id,
                current_rel_path,
                True,
                ref_key,
                link_context(markdown, *match.span()),
            )
        )
    for match in RE_BARE_URL.finditer(markdown):
        start, end = match.span()
        if any(start < used_end and end > used_start for used_start, used_end in occupied_ranges):
            continue
        edge_index += 1
        target = match.group(0).rstrip(".,;:")
        edges.append(
            build_edge(
                edge_index,
                source_section_id,
                source_heading_path,
                target,
                target,
                current_page_id,
                current_rel_path,
                False,
                None,
                link_context(markdown, start, end),
            )
        )
    return edges


def iter_inline_link_matches(markdown: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    index = 0
    in_fence = False
    while index < len(markdown):
        if markdown.startswith("```", index):
            in_fence = not in_fence
            index += 3
            continue
        if in_fence:
            index += 1
            continue
        if markdown[index] != "[" or (index > 0 and markdown[index - 1] == "!"):
            index += 1
            continue
        close_bracket = markdown.find("]", index + 1)
        if close_bracket == -1 or close_bracket + 1 >= len(markdown) or markdown[close_bracket + 1] != "(":
            index += 1
            continue
        close_paren = find_balanced_link_target_end(markdown, close_bracket + 2)
        if close_paren is None:
            index += 1
            continue
        matches.append(
            {
                "start": index,
                "end": close_paren + 1,
                "text": markdown[index + 1 : close_bracket],
                "target": markdown[close_bracket + 2 : close_paren],
            }
        )
        index = close_paren + 1
    return matches


def find_balanced_link_target_end(markdown: str, start: int) -> int | None:
    depth = 0
    index = start
    while index < len(markdown):
        char = markdown[index]
        if char == "\\":
            index += 2
            continue
        if char == "\n":
            return None
        if char == "(":
            depth += 1
        elif char == ")":
            if depth == 0:
                return index
            depth -= 1
        index += 1
    return None


def link_context(markdown: str, start: int, end: int, *, window: int = 180) -> str:
    left = max(0, start - window)
    right = min(len(markdown), end + window)
    while left > 0 and markdown[left - 1] not in ".!?\n":
        left -= 1
    while right < len(markdown) and markdown[right] not in ".!?\n":
        right += 1
    context = markdown[left:right].strip()
    return truncate_context(re.sub(r"\s+", " ", inline_markdown_to_text(context)).strip())


def truncate_context(value: str, limit: int = 320) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def build_edge(
    edge_index: int,
    section_id: str,
    heading_path: list[str],
    link_text: str,
    raw_target: str,
    page_id: str,
    rel_path: str,
    is_reference: bool,
    reference_key: str | None,
    context: str = "",
) -> dict[str, Any]:
    link_type, resolved, target_page_id, target_anchor = classify_and_resolve_link(raw_target, current_page_id=page_id, current_rel_path=rel_path)
    return {
        "edge_id": f"{section_id}:{edge_index}",
        "source_section_id": section_id,
        "source_heading_path": heading_path,
        "link_text": link_text.strip(),
        "url_raw": raw_target,
        "url_resolved": resolved,
        "link_type": link_type,
        "target_page_id": target_page_id,
        "target_anchor": target_anchor,
        "is_reference_link": is_reference,
        "reference_key": reference_key,
        "context": context,
    }


def classify_and_resolve_link(raw_target: str, *, current_page_id: str, current_rel_path: str) -> tuple[str, str | None, str | None, str | None]:
    target = raw_target.strip()
    if not target:
        return "other", None, None, None
    if target.startswith("mailto:"):
        return "mailto", target, None, None
    if target.startswith("#"):
        anchor = normalize_anchor(target[1:])
        return "wiki_section", f"/wiki/{current_page_id}#{anchor}" if anchor else f"/wiki/{current_page_id}", current_page_id, anchor
    if target.startswith("/wiki/"):
        wiki_path, anchor = split_anchor(target[len("/wiki/") :])
        wiki_path = wiki_path.strip("/")
        if looks_like_asset(wiki_path):
            return "wiki_asset", f"/wiki/{wiki_path}", None, anchor
        resolved = f"/wiki/{wiki_path}" if wiki_path else None
        if anchor:
            return "wiki_section", f"{resolved}#{anchor}" if resolved else None, wiki_path or None, anchor
        return "wiki_article", resolved, wiki_path or None, None
    if target.startswith("http://") or target.startswith("https://"):
        parsed = urlsplit(target)
        host = (parsed.hostname or "").lower()
        fragment = normalize_anchor(parsed.fragment)
        if host in PPY_HOSTS:
            path = parsed.path or ""
            if path.startswith("/wiki/"):
                wiki_path = path[len("/wiki/") :].strip("/")
                if looks_like_asset(wiki_path):
                    return "wiki_asset", target, None, fragment
                return ("wiki_section" if fragment else "wiki_article"), target, wiki_path or None, fragment
            return "ppy_internal", target, None, None
        return "external", target, None, None
    rel_path, anchor = split_anchor(target)
    rel_norm = resolve_relative_wiki_path(current_rel_path, rel_path)
    if rel_norm:
        if looks_like_asset(rel_norm):
            return "wiki_asset", rel_norm, None, anchor
        resolved = f"/wiki/{rel_norm}"
        if anchor:
            return "wiki_section", f"{resolved}#{anchor}", rel_norm, anchor
        return "wiki_article", resolved, rel_norm, None
    return "other", target, None, None


def extract_images(*, markdown: str, current_page_id: str, current_rel_path: str, link_defs: dict[str, str]) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    def add_image(alt: str, target: str, title: str, is_ref: bool, ref_key: str | None) -> None:
        raw_target = clean_link_target(target)
        link_type, resolved, target_page_id, target_anchor = classify_and_resolve_link(raw_target, current_page_id=current_page_id, current_rel_path=current_rel_path)
        if link_type not in {"wiki_asset", "external", "ppy_internal"}:
            return
        key = (alt.strip(), raw_target, title.strip())
        if key in seen:
            return
        seen.add(key)
        images.append(
            {
                "image_id": f"img{len(images) + 1}",
                "alt": alt.strip(),
                "title": title.strip(),
                "url_raw": raw_target,
                "url_resolved": resolved,
                "link_type": link_type,
                "target_page_id": target_page_id,
                "target_anchor": target_anchor,
                "is_reference_image": is_ref,
                "reference_key": ref_key,
            }
        )

    for match in RE_MD_IMAGE_INLINE.finditer(markdown):
        add_image(match.group(1) or "", match.group(2) or "", match.group(3) or "", False, None)
    for match in RE_MD_IMAGE_REF.finditer(markdown):
        ref_key = (match.group(2) or "").strip()
        add_image(match.group(1) or "", link_defs.get(ref_key.lower(), ref_key), "", True, ref_key)
    return images


def extract_formulas(markdown: str) -> list[dict[str, str]]:
    formulas: list[dict[str, str]] = []
    seen: set[str] = set()
    for source, expressions in [
        ("inline_code", [match.group(1).strip() for match in RE_INLINE_CODE.finditer(markdown)]),
        ("code_block", [line.strip() for match in RE_FENCED_BLOCK.finditer(markdown) for line in match.group(1).splitlines()]),
    ]:
        for expression in expressions:
            if not looks_like_formula(expression) or expression in seen:
                continue
            seen.add(expression)
            formulas.append({"formula_id": f"f{len(formulas) + 1}", "source": source, "expression": expression})
    return formulas


def markdown_to_text(markdown: str) -> str:
    text = RE_CONTAINER_MARKER_LINE.sub("", markdown)
    text = RE_ID_MARKER_LINE.sub("", text)
    text = RE_DIRECTIVE.sub("", text)
    text = RE_MD_IMAGE_INLINE.sub(lambda match: image_placeholder(match.group(1), match.group(3)), text)
    text = RE_MD_IMAGE_REF.sub(r"[Image: \1]", text)
    text = RE_INLINE_LINK.sub(r"\1", text)
    text = RE_REFERENCE_LINK.sub(r"\1", text)
    text = RE_FOOTNOTE_REF.sub("", text)
    text = text.replace("\\\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_preview_paragraph(body: str) -> str:
    text = RE_FENCED_BLOCK_ANY.sub("", body)
    for block in re.split(r"\n\s*\n", text):
        value = block.strip()
        if not value or value.startswith("---") or RE_MD_IMAGE_INLINE.fullmatch(value) or RE_MD_IMAGE_REF.fullmatch(value):
            continue
        cleaned = inline_markdown_to_text(" ".join(line.strip() for line in value.splitlines()))
        if cleaned:
            return cleaned
    return ""


def extract_banner_image_from_sections(sections: list[dict[str, Any]]) -> dict[str, Any] | None:
    for section in sections:
        images = section.get("images", [])
        if images:
            image = dict(images[0])
            image["section_id"] = section.get("section_id")
            image["section_title"] = section.get("title")
            return image
    return None


def strip_heading_custom_id(title: str) -> tuple[str, str | None]:
    match = RE_HEADING_CUSTOM_ID.match(title)
    if not match:
        return title, None
    return match.group(1).strip(), match.group(2).strip() or None


def infer_title_from_body(markdown: str, fallback: str) -> str:
    for line in markdown.splitlines():
        match = RE_HEADING.match(line)
        if match and len(match.group(1)) == 1:
            return strip_heading_custom_id(match.group(2).strip())[0] or fallback
    return fallback


def normalize_tags(raw_tags: Any) -> list[str]:
    return [str(tag).strip() for tag in raw_tags if str(tag).strip()] if isinstance(raw_tags, list) else []


def normalize_series(raw_series: Any) -> list[str]:
    if isinstance(raw_series, list):
        values = [str(item).strip() for item in raw_series]
    elif raw_series is None:
        values = []
    else:
        raw = str(raw_series).strip()
        values = [part.strip() for part in raw.split(",")] if "," in raw else [raw]
    return list(dict.fromkeys(value for value in values if value))


def parse_news_date(value: str) -> str | None:
    raw = value.strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S %z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).isoformat()
        except ValueError:
            continue
    return None


def clean_scalar(value: Any) -> str:
    return "" if value is None else str(value).strip()


def derive_style_issues(style: dict[str, bool]) -> list[str]:
    checks = [
        ("layout_is_post", "layout_not_post", True),
        ("has_series", "missing_series", True),
        ("date_parsed", "invalid_frontmatter_date", True),
        ("has_preview_paragraph", "missing_preview_paragraph", True),
        ("has_banner_image", "missing_banner_image", True),
        ("no_h1_in_body", "has_h1_heading", True),
        ("no_footnotes_in_body", "contains_footnotes", True),
        ("title_has_no_markdown", "title_contains_markdown", True),
    ]
    issues = [issue for key, issue, expected in checks if style.get(key) is not expected]
    if style.get("has_banner_image") and not style.get("banner_alt_is_empty"):
        issues.append("banner_alt_not_empty")
    return issues


def inline_markdown_to_text(text: str) -> str:
    out = RE_MD_IMAGE_INLINE.sub("", text)
    out = RE_MD_IMAGE_REF.sub("", out)
    out = RE_INLINE_LINK.sub(r"\1", out)
    out = RE_REFERENCE_LINK.sub(r"\1", out)
    out = RE_DIRECTIVE.sub("", out)
    out = RE_FOOTNOTE_REF.sub("", out)
    out = re.sub(r"(?<!\\)[*_~`]", "", out)
    return re.sub(r"\s+", " ", out).strip()


def strip_definition_lines(markdown: str) -> str:
    kept: list[str] = []
    in_fence = False
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            kept.append(line)
            continue
        if in_fence or (not RE_FOOTNOTE_DEF.match(stripped) and not RE_LINK_DEF.match(stripped)):
            kept.append(line)
    return "\n".join(kept).strip()


def extract_directives(markdown: str) -> list[dict[str, str]]:
    return [{"key": match.group(1).strip().lower(), "value": match.group(2).strip()} for match in RE_DIRECTIVE.finditer(markdown)]


def extract_container_directives(markdown: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    in_fence = False
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = RE_CONTAINER_MARKER_LINE.match(line)
        if match:
            out.append({"key": "container", "value": (match.group(1) or "end").strip()})
    return out


def strip_container_markers(markdown: str) -> str:
    kept: list[str] = []
    in_fence = False
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            kept.append(line)
            continue
        if in_fence or not RE_CONTAINER_MARKER_LINE.match(line):
            kept.append(line)
    return "\n".join(kept).strip()


def strip_code_for_semantic_extraction(markdown: str) -> str:
    return RE_INLINE_CODE.sub("", RE_FENCED_BLOCK_ANY.sub("", markdown))


def extract_inline_anchor_ids(markdown: str) -> list[str]:
    return [anchor for match in RE_ID_MARKER_LINE.finditer(markdown) if (anchor := normalize_anchor(match.group(1)))]


def strip_inline_anchor_ids(markdown: str) -> str:
    return RE_ID_MARKER_LINE.sub("", markdown).strip()


def is_table_header_and_separator(header: str, separator: str) -> bool:
    if "|" not in header or "|" not in separator:
        return False
    cells = split_table_cells(separator)
    return len(cells) >= 2 and all(re.match(r"^:?-+:?$", cell.strip()) for cell in cells if cell.strip())


def looks_like_table_row(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped and "|" in stripped and not stripped.startswith(("#", ">", "```")) and len(split_table_cells(stripped)) >= 2)


def split_table_cells(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    cells: list[str] = []
    buffer: list[str] = []
    escaped = False
    for char in stripped:
        if escaped:
            buffer.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "|":
            cells.append("".join(buffer).strip())
            buffer = []
        else:
            buffer.append(char)
    cells.append("".join(buffer).strip())
    return cells


def normalize_row(cells: list[str], width: int) -> list[str]:
    if len(cells) < width:
        return cells + [""] * (width - len(cells))
    if len(cells) > width:
        return cells[: width - 1] + [" | ".join(cells[width - 1 :]).strip()]
    return cells


def parse_table_alignments(cells: list[str]) -> list[str]:
    out: list[str] = []
    for cell in cells:
        stripped = cell.strip()
        if stripped.startswith(":") and stripped.endswith(":"):
            out.append("center")
        elif stripped.startswith(":"):
            out.append("left")
        elif stripped.endswith(":"):
            out.append("right")
        else:
            out.append("default")
    return out


def tables_to_text(tables: list[dict[str, Any]], max_rows: int = 40) -> str:
    lines: list[str] = []
    for index, table in enumerate(tables, start=1):
        headers = table.get("headers", [])
        lines.append(f"Table {index}: columns: {', '.join(headers)}")
        for row_index, row in enumerate(table.get("rows", [])[:max_rows], start=1):
            pairs = [f"{headers[idx] if idx < len(headers) else f'col_{idx + 1}'}={cell}" for idx, cell in enumerate(row) if str(cell).strip()]
            if pairs:
                lines.append(f"Row {row_index}: " + "; ".join(pairs))
    return "\n".join(lines).strip()


def formulas_to_text(formulas: list[dict[str, str]]) -> str:
    return "\n".join(["Formulas:", *(f"- {item['expression']}" for item in formulas)]).strip() if formulas else ""


def looks_like_formula(expression: str) -> bool:
    value = expression.strip()
    if not 3 <= len(value) <= 300:
        return False
    if "http://" in value or "https://" in value:
        return False
    if value.startswith(("!", "/", "13:")):
        return False
    if "<" in value and ">" in value and "=" not in value:
        return False
    return any(op in value for op in ["=", "+", "-", "*", "/", "^", "<", ">"]) and bool(re.search(r"[A-Za-z0-9]", value))


def resolve_relative_wiki_path(current_rel_path: str, rel_target: str) -> str | None:
    target = rel_target.strip()
    if not target:
        return None
    candidate = posixpath.normpath(posixpath.join(posixpath.dirname(current_rel_path), target)).replace("\\", "/")
    if candidate.endswith(".md"):
        candidate = candidate[: -len(".md")]
    if candidate.endswith("/en"):
        candidate = candidate[: -len("/en")]
    if candidate == ".":
        candidate = posixpath.dirname(current_rel_path)
    return None if candidate.startswith("../") else candidate.strip("/")


def split_anchor(value: str) -> tuple[str, str | None]:
    if "#" not in value:
        return value, None
    path, anchor = value.split("#", 1)
    return path, normalize_anchor(anchor)


def normalize_anchor(anchor: str) -> str | None:
    value = anchor.strip()
    return value.lower().replace(" ", "-") if value else None


def looks_like_asset(path: str) -> bool:
    lower = path.lower()
    return lower.startswith("shared/") or any(lower.endswith(ext) for ext in ASSET_EXTENSIONS)


def clean_link_target(target: str) -> str:
    value = target.strip()
    if " " in value and not value.startswith("<"):
        value = value.split(" ", 1)[0]
    return value.strip("<>")


def image_placeholder(alt: str, title: str | None) -> str:
    label = (alt or "").strip() or (title or "").strip() or "Image"
    return f"[Image: {label}]"


def slugify(text: str) -> str:
    value = text.strip().lower()
    value = re.sub(r"[^a-z0-9!_\- ]+", "", value)
    value = re.sub(r"\s+", "-", value)
    return value.strip("-") or "section"
