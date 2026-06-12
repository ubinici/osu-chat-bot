from __future__ import annotations

from typing import Any, Iterable
import re

from ..domain.models import Chunk


def build_chunks_from_record(
    record: dict[str, Any],
    *,
    max_table_rows: int = 40,
    max_intro_chars: int = 1400,
    max_chunk_chars: int = 7000,
) -> list[Chunk]:
    root_id, root_kind = resolve_root_id(record)
    source_type = "news" if record.get("source") == "osu-news" else "wiki"
    title = str(record.get("title") or root_id)
    base = build_base_metadata(record, root_id=root_id, root_kind=root_kind, source_type=source_type, title=title)
    chunks: list[Chunk] = []

    article_text = build_article_text(record, title=title, max_intro_chars=max_intro_chars)
    chunks.append(make_chunk(record, base, article_text, "article", len(chunks)))

    for section in safe_dicts(record.get("sections")):
        section_id = str(section.get("section_id") or "").strip()
        section_text = str(section.get("text") or "").strip()
        if section_id and section_text:
            section_parts = split_text_blocks(section_text, max_chars=max_chunk_chars - 300)
            metadata = {
                **base,
                "chunk_type": "section",
                "section_id": section_id,
                "parent_section_id": section.get("parent_section_id"),
                "section_title": section.get("title"),
                "heading_path": safe_str_list(section.get("heading_path")),
                "heading_path_str": " > ".join(safe_str_list(section.get("heading_path"))),
                "section_level": section.get("level"),
                "table_count": section.get("table_count"),
                "formula_count": section.get("formula_count"),
                "image_count": section.get("image_count"),
                "char_count": section.get("char_count"),
                "word_count": section.get("word_count"),
            }
            for part_index, section_part in enumerate(section_parts, start=1):
                part_metadata = metadata
                if len(section_parts) > 1:
                    part_metadata = {
                        **metadata,
                        "chunk_part_index": part_index,
                        "chunk_part_count": len(section_parts),
                        "original_char_count": len(section_text),
                    }
                chunks.append(
                    make_chunk(
                        record,
                        part_metadata,
                        build_section_text(
                            title,
                            str(section.get("title") or section_id),
                            safe_str_list(section.get("heading_path")),
                            section_part,
                        ),
                        "section",
                        len(chunks),
                        section=section,
                    )
                )
        for table in safe_dicts(section.get("tables")):
            table_id = str(table.get("table_id") or f"table-{len(chunks) + 1}")
            table_parts = build_table_text_parts(
                title,
                str(section.get("title") or section_id),
                table,
                max_rows=max_table_rows,
                max_chars=max_chunk_chars,
            )
            for part_index, table_part in enumerate(table_parts, start=1):
                chunks.append(
                    make_chunk(
                        record,
                        {
                            **base,
                            "chunk_type": "table",
                            "section_id": section_id,
                            "section_title": section.get("title"),
                            "table_id": table_id,
                            "column_count": table.get("column_count"),
                            "row_count": table.get("row_count"),
                            "table_headers": safe_str_list(table.get("headers")),
                            "chunk_part_index": part_index if len(table_parts) > 1 else None,
                            "chunk_part_count": len(table_parts) if len(table_parts) > 1 else None,
                        },
                        table_part,
                        "table",
                        len(chunks),
                        section=section,
                    )
                )
        for formula in safe_dicts(section.get("formulae")):
            expression = str(formula.get("expression") or "").strip()
            if expression:
                chunks.append(
                    make_chunk(
                        record,
                        {
                            **base,
                            "chunk_type": "formula",
                            "section_id": section_id,
                            "section_title": section.get("title"),
                            "formula_id": formula.get("formula_id"),
                            "formula_source": formula.get("source"),
                        },
                        f"Article: {title}\nSection: {section.get('title') or section_id}\nFormula ({formula.get('source', 'unknown')}): {expression}",
                        "formula",
                        len(chunks),
                        section=section,
                    )
                )

    for citation in safe_dicts(record.get("citations")):
        citation_id = str(citation.get("citation_id") or "").strip()
        citation_text = str(citation.get("text") or "").strip()
        if citation_id and citation_text:
            chunks.append(
                make_chunk(
                    record,
                    {
                        **base,
                        "chunk_type": "citation",
                        "citation_id": citation_id,
                        "citation_missing_definition": bool(citation.get("missing_definition", False)),
                    },
                    f"Article: {title}\nCitation {citation_id}: {citation_text}",
                    "citation",
                    len(chunks),
                )
            )

    return [chunk for chunk in chunks if chunk.text.strip()]


def iter_chunks_from_records(records: Iterable[dict[str, Any]]) -> Iterable[Chunk]:
    for record in records:
        yield from build_chunks_from_record(record)


def make_chunk(record: dict[str, Any], metadata: dict[str, Any], text: str, chunk_type: str, index: int, section: dict[str, Any] | None = None) -> Chunk:
    root_id = str(metadata["root_id"])
    source_type = str(metadata["source_type"])
    heading_path = safe_str_list((section or {}).get("heading_path"))
    section_id = str((section or {}).get("section_id") or "")
    suffix = section_id or chunk_type
    chunk_id = f"{root_id}::{chunk_type}::{suffix}" if chunk_type != "article" else f"{root_id}::article"
    part_index = metadata.get("chunk_part_index")
    if part_index:
        chunk_id = f"{chunk_id}::part-{part_index}"
    if chunk_type not in {"article", "section"}:
        chunk_id = f"{chunk_id}::{index}"
    return Chunk(
        id=chunk_id,
        document_id=root_id,
        source_type=source_type,
        file_path=str(record.get("repo_rel_path") or ""),
        osu_url=str(record.get("osu_url") or ""),
        title=str(record.get("title") or root_id),
        text=text.strip(),
        chunk_index=index,
        heading_path=heading_path or [str(record.get("title") or root_id)],
        tags=safe_str_list(record.get("tags")) + safe_str_list(record.get("series_tags")),
        date=record.get("date_iso") or record.get("filename_date"),
        series=record.get("series_primary"),
        metadata=compact_metadata({**metadata, "chunk_type": chunk_type}),
    )


def build_base_metadata(record: dict[str, Any], *, root_id: str, root_kind: str, source_type: str, title: str) -> dict[str, Any]:
    metadata = {
        "chunk_schema": "hier-v1",
        "source": record.get("source"),
        "source_type": source_type,
        "root_id": root_id,
        "root_kind": root_kind,
        "repo_rel_path": record.get("repo_rel_path"),
        "osu_url": record.get("osu_url"),
        "domain": record.get("domain"),
        "subculture": record.get("subculture"),
        "lang": record.get("lang"),
        "title": title,
        "topic": record.get("topic"),
        "year": record.get("year"),
        "series_primary": record.get("series_primary"),
        "series_tags": safe_str_list(record.get("series_tags")),
        "is_stub": record.get("is_stub"),
        "page_id": record.get("page_id"),
        "post_id": record.get("post_id"),
    }
    return compact_metadata(metadata)


def build_article_text(record: dict[str, Any], *, title: str, max_intro_chars: int) -> str:
    lines = [f"Title: {title}"]
    for label, key in [("Domain", "domain"), ("Subculture", "subculture"), ("Year", "year"), ("Series", "series_primary")]:
        value = str(record.get(key) or "").strip()
        if value:
            lines.append(f"{label}: {value}")
    preview = str(record.get("preview_paragraph") or "").strip()
    intro = extract_intro_text(record.get("sections"), max_chars=max_intro_chars)
    if preview:
        lines.extend(["", "Preview:", preview])
    if intro and intro != preview:
        lines.extend(["", "Intro excerpt:", intro])
    return "\n".join(lines).strip()


def extract_intro_text(sections: Any, max_chars: int) -> str:
    for section in safe_dicts(sections):
        text = str(section.get("text") or "").strip()
        if text:
            return text[:max_chars].rstrip() + "..." if len(text) > max_chars else text
    return ""


def build_section_text(title: str, section_title: str, heading_path: list[str], section_text: str) -> str:
    lines = [f"Article: {title}", f"Section: {section_title}"]
    if heading_path:
        lines.append(f"Heading path: {' > '.join(heading_path)}")
    lines.extend(["", section_text])
    return "\n".join(lines).strip()


def build_table_text(title: str, section_title: str, table: dict[str, Any], max_rows: int) -> str:
    parts = build_table_text_parts(title, section_title, table, max_rows=max_rows, max_chars=7000)
    return parts[0] if parts else ""


def build_table_text_parts(title: str, section_title: str, table: dict[str, Any], *, max_rows: int, max_chars: int) -> list[str]:
    headers = safe_str_list(table.get("headers"))
    rows = table.get("rows", [])
    if not isinstance(rows, list):
        rows = []
    prefix = [f"Article: {title}", f"Section: {section_title}"]
    if headers:
        prefix.append(f"Columns: {', '.join(headers)}")
    prefix.append("")

    parts: list[str] = []
    part_lines = prefix.copy()
    rows_in_part = 0

    def flush() -> None:
        nonlocal part_lines, rows_in_part
        if rows_in_part:
            parts.append("\n".join(part_lines).strip())
        part_lines = prefix.copy()
        rows_in_part = 0

    for index, row in enumerate(rows, start=1):
        if not isinstance(row, list):
            continue
        pairs = []
        for idx, cell in enumerate(row):
            value = str(cell).strip()
            if value:
                pairs.append(f"{headers[idx] if idx < len(headers) else f'col_{idx + 1}'}={value}")
        if pairs:
            row_line = f"Row {index}: " + "; ".join(pairs)
            projected = "\n".join([*part_lines, row_line]).strip()
            if rows_in_part and (rows_in_part >= max_rows or len(projected) > max_chars):
                flush()
            if len(row_line) > max_chars:
                for row_part in split_text_blocks(row_line, max_chars=max_chars - len("\n".join(prefix)) - 20):
                    part_lines.append(row_part)
                    rows_in_part = 1
                    flush()
                continue
            part_lines.append(row_line)
            rows_in_part += 1
    flush()
    return parts


def split_text_blocks(text: str, *, max_chars: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    max_chars = max(1200, max_chars)
    if len(text) <= max_chars:
        return [text]

    blocks = _paragraph_blocks(text)
    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for block in blocks:
        block_parts = _split_long_block(block, max_chars=max_chars) if len(block) > max_chars else [block]
        for block_part in block_parts:
            separator_len = 2 if current else 0
            if current and current_len + separator_len + len(block_part) > max_chars:
                parts.append("\n\n".join(current).strip())
                current = []
                current_len = 0
            current.append(block_part)
            current_len += (2 if current_len else 0) + len(block_part)
    if current:
        parts.append("\n\n".join(current).strip())
    return [part for part in parts if part]


def _paragraph_blocks(text: str) -> list[str]:
    return [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]


def _split_long_block(block: str, *, max_chars: int) -> list[str]:
    lines = [line.rstrip() for line in block.splitlines() if line.strip()]
    if len(lines) > 1:
        return split_text_blocks("\n\n".join(lines), max_chars=max_chars)

    words = block.split()
    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        if current and current_len + 1 + len(word) > max_chars:
            parts.append(" ".join(current))
            current = []
            current_len = 0
        if len(word) > max_chars:
            if current:
                parts.append(" ".join(current))
                current = []
                current_len = 0
            parts.extend(word[index : index + max_chars] for index in range(0, len(word), max_chars))
            continue
        current.append(word)
        current_len += (1 if current_len else 0) + len(word)
    if current:
        parts.append(" ".join(current))
    return parts


def resolve_root_id(record: dict[str, Any]) -> tuple[str, str]:
    for key, kind in [("page_id", "page"), ("post_id", "post"), ("id", "record"), ("repo_rel_path", "record")]:
        value = str(record.get(key) or "").strip()
        if value:
            return value, kind
    return "unknown", "record"


def build_chunks(record: dict[str, Any]) -> list[Chunk]:
    return build_chunks_from_record(record)


def safe_dicts(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def safe_str_list(value: Any) -> list[str]:
    return [str(item).strip() for item in value if str(item).strip()] if isinstance(value, list) else []


def compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metadata.items() if value is not None}
