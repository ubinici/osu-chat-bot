from __future__ import annotations

from typing import Any, Iterable

from ..domain.models import Chunk


DEFAULT_MAX_CHUNK_TOKENS = 180
DEFAULT_CHUNK_OVERLAP_TOKENS = 24


def build_chunks_from_record(
    record: dict[str, Any],
    *,
    max_table_rows: int = 40,
    max_intro_chars: int = 1400,
    max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
) -> list[Chunk]:
    if max_chunk_tokens <= 0:
        raise ValueError("max_chunk_tokens must be greater than zero.")
    if chunk_overlap_tokens < 0 or chunk_overlap_tokens >= max_chunk_tokens:
        raise ValueError("chunk_overlap_tokens must be between zero and max_chunk_tokens.")
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
            section_parts = split_text_blocks(
                section_text,
                max_tokens=max_chunk_tokens,
                overlap_tokens=chunk_overlap_tokens,
            )
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
                        "original_token_count": token_count(section_text),
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
                max_tokens=max_chunk_tokens,
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
    parts = build_table_text_parts(
        title,
        section_title,
        table,
        max_rows=max_rows,
        max_tokens=DEFAULT_MAX_CHUNK_TOKENS,
    )
    return parts[0] if parts else ""


def build_table_text_parts(
    title: str,
    section_title: str,
    table: dict[str, Any],
    *,
    max_rows: int,
    max_tokens: int,
) -> list[str]:
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
            if rows_in_part and (rows_in_part >= max_rows or token_count(projected) > max_tokens):
                flush()
            available_tokens = max(1, max_tokens - token_count("\n".join(prefix)))
            if token_count(row_line) > available_tokens:
                for row_part in split_text_blocks(row_line, max_tokens=available_tokens):
                    part_lines.append(row_part)
                    rows_in_part = 1
                    flush()
                continue
            part_lines.append(row_line)
            rows_in_part += 1
    flush()
    return parts


def split_text_blocks(text: str, *, max_tokens: int, overlap_tokens: int = 0) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero.")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be between zero and max_tokens.")

    tokens = text.split()
    if len(tokens) <= max_tokens:
        return [text]

    step = max_tokens - overlap_tokens
    parts: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        parts.append(" ".join(tokens[start:end]))
        if end == len(tokens):
            break
        start += step
    return parts


def token_count(text: str) -> int:
    """Cheap model-independent token estimate used to cap embedding inputs."""

    return len(text.split())


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
