from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re

from ..corpus.markdown import strip_markdown
from ..domain.artifacts import DOCUMENTS_FILE, TERMS_FILE, load_records, write_json
from ..domain.models import Entity

COMMON_OSU_TERMS = {
    "osu!",
    "osu!standard",
    "osu!taiko",
    "osu!catch",
    "osu!mania",
    "osu!direct",
    "osu!supporter",
    "osu!team",
    "osu!lazer",
    "osu! World Cup",
    "osu!taiko World Cup",
    "osu!catch World Cup",
    "osu!mania World Cup",
    "lazer",
    "BanchoBot",
    "pp",
    "performance points",
    "NAT",
    "BN",
    "GMT",
    "QAT",
}

STOPWORD_ALIASES = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "was",
    "were",
    "will",
    "would",
    "can",
    "could",
    "should",
    "has",
    "have",
    "had",
    "not",
    "game",
    "games",
}


def build_terms_from_artifacts(artifact_dir: Path) -> list[Entity]:
    documents = load_records(artifact_dir / DOCUMENTS_FILE)
    entities = build_terms(documents)
    write_json(artifact_dir / TERMS_FILE, [entity.__dict__ for entity in entities])
    write_json(
        artifact_dir / "terms_report.json",
        {
            "entities": len(entities),
            "aliases": sum(len(entity.aliases) for entity in entities),
            "sources": "documents_structured.jsonl",
        },
    )
    return entities


def build_terms(documents: list[dict]) -> list[Entity]:
    aliases_by_canonical: dict[str, set[str]] = defaultdict(set)
    sources_by_canonical: dict[str, set[str]] = defaultdict(set)

    for term in COMMON_OSU_TERMS:
        add_term(aliases_by_canonical, sources_by_canonical, term, "common", term)

    for document in documents:
        title = str(document.get("title") or "")
        source_path = str(document.get("repo_rel_path") or document.get("page_id") or document.get("post_id") or "")
        add_term(aliases_by_canonical, sources_by_canonical, title, source_path, title)
        for identifier_key in ("page_id", "post_id", "topic", "domain", "subculture", "slug"):
            value = str(document.get(identifier_key) or "")
            if value:
                add_term(aliases_by_canonical, sources_by_canonical, value, source_path, value)
        for heading in _section_titles(document):
            add_term(aliases_by_canonical, sources_by_canonical, heading, source_path, heading)
        for tag in [*_list(document.get("tags")), *_list(document.get("series_tags"))]:
            add_term(aliases_by_canonical, sources_by_canonical, tag, source_path, tag)
        for part in _path_terms(source_path):
            add_term(aliases_by_canonical, sources_by_canonical, part, source_path, part)
        for edge in _list_of_dicts(document.get("link_edges")):
            text = str(edge.get("link_text") or "")
            target_page_id = str(edge.get("target_page_id") or "")
            if target_page_id:
                canonical_target = target_page_id.rsplit("/", 1)[-1]
                if link_alias_matches_target(text, canonical_target):
                    add_term(aliases_by_canonical, sources_by_canonical, text, source_path, canonical_target)
                add_term(aliases_by_canonical, sources_by_canonical, canonical_target, source_path, canonical_target)

    aliases_by_canonical, sources_by_canonical = merge_canonical_variants(aliases_by_canonical, sources_by_canonical)

    entities = []
    for canonical, aliases in aliases_by_canonical.items():
        cleaned_aliases = sorted({alias for alias in aliases if _is_useful_term(alias)}, key=str.lower)
        if not cleaned_aliases or not _is_useful_term(canonical):
            continue
        entities.append(
            Entity(
                canonical=canonical,
                aliases=cleaned_aliases,
                sources=sorted(sources_by_canonical[canonical])[:20],
                score=min(4.0, 1.0 + len(cleaned_aliases) / 10),
            )
        )
    return sorted(entities, key=lambda entity: (entity.canonical.lower(), entity.canonical))


def merge_canonical_variants(
    aliases_by_canonical: dict[str, set[str]],
    sources_by_canonical: dict[str, set[str]],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    canonical_by_key: dict[str, str] = {}
    merged_aliases: dict[str, set[str]] = defaultdict(set)
    merged_sources: dict[str, set[str]] = defaultdict(set)

    for canonical in sorted(aliases_by_canonical, key=lambda value: (value.casefold(), value)):
        key = canonical.casefold()
        display = canonical_by_key.setdefault(key, canonical)
        merged_aliases[display].update(aliases_by_canonical[canonical])
        merged_aliases[display].add(canonical)
        merged_sources[display].update(sources_by_canonical.get(canonical, set()))

    return merged_aliases, merged_sources


def add_term(
    aliases_by_canonical: dict[str, set[str]],
    sources_by_canonical: dict[str, set[str]],
    raw_alias: str,
    source: str,
    canonical_hint: str,
) -> None:
    alias = normalize_term(raw_alias)
    canonical = normalize_term(canonical_hint) or alias
    if not alias or not canonical:
        return
    aliases_by_canonical[canonical].add(alias)
    for variant in osu_specific_alias_variants(alias):
        aliases_by_canonical[canonical].add(variant)
    plural = plural_alias(alias)
    if plural:
        aliases_by_canonical[canonical].add(plural)
    sources_by_canonical[canonical].add(source)
    acronym = acronym_for(alias)
    if acronym:
        aliases_by_canonical[canonical].add(acronym)


def normalize_term(value: str) -> str:
    value = strip_markdown(value or "")
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def acronym_for(value: str) -> str | None:
    words = re.findall(r"[A-Za-z][A-Za-z0-9!]+", value)
    if len(words) < 2 or len(words) > 6:
        return None
    acronym = "".join(word[0] for word in words).upper()
    if acronym.casefold() in STOPWORD_ALIASES:
        return None
    return acronym if len(acronym) >= 2 else None


def plural_alias(value: str) -> str | None:
    if not re.fullmatch(r"[A-Za-z0-9 ]+", value):
        return None
    if value.casefold().endswith("s"):
        return None
    if len(value) < 4:
        return None
    return f"{value}s"


def osu_specific_alias_variants(value: str) -> set[str]:
    if not value.casefold().startswith("osu!"):
        return set()
    variants = {value.replace("osu!", "osu")}
    if len(value) > len("osu!") and not value[len("osu!")].isspace():
        variants.add(value.replace("osu!", "osu! ", 1))
        variants.add(value.replace("osu!", "osu ", 1))
    return {normalize_term(variant) for variant in variants if normalize_term(variant)}


def link_alias_matches_target(alias: str, target: str) -> bool:
    alias_norm = normalize_term(alias).casefold()
    target_norm = normalize_term(target).casefold()
    if not alias_norm or not target_norm:
        return False
    if alias_norm == target_norm:
        return True
    if alias_norm in target_norm or target_norm in alias_norm:
        return len(alias_norm) >= 4 and len(target_norm) >= 4
    return acronym_for(alias_norm) == target_norm.upper() or acronym_for(target_norm) == alias_norm.upper()


def detect_entities(query: str, entities: list[Entity]) -> list[Entity]:
    normalized = query.casefold()
    matches: list[tuple[Entity, tuple[int, int], str]] = []
    for entity in entities:
        for alias in entity.aliases:
            alias_key = alias.casefold()
            if alias_key in STOPWORD_ALIASES:
                continue
            span = _alias_span_in_query(alias_key, normalized)
            if span:
                matches.append((entity, span, alias_key))
                break
    return _select_entity_matches(matches)


def _path_terms(file_path: str) -> list[str]:
    path = Path(file_path)
    return [part for part in path.parts if part not in {"database", "osu-wiki", "wiki", "news", "en.md"}]


def _is_useful_term(value: str) -> bool:
    if not value:
        return False
    if len(value) < 2:
        return False
    if len(value) > 80:
        return False
    if re.fullmatch(r"\d{4}", value):
        return False
    if re.fullmatch(r"\d+(st|nd|rd|th) place", value.casefold()):
        return False
    if value.casefold() in {"img", "image", "main article", "click here", "game", "games"}:
        return False
    if value.casefold() in STOPWORD_ALIASES:
        return False
    return bool(re.search(r"[A-Za-z0-9!]", value))


def _alias_in_query(alias: str, query: str) -> bool:
    return _alias_span_in_query(alias, query) is not None


def _alias_span_in_query(alias: str, query: str) -> tuple[int, int] | None:
    if re.fullmatch(r"[a-z0-9 ]+", alias):
        match = re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", query)
        return match.span() if match else None
    match = re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", query)
    return match.span() if match else None


def _select_entity_matches(matches: list[tuple[Entity, tuple[int, int], str]]) -> list[Entity]:
    selected: list[tuple[Entity, tuple[int, int], str]] = []
    seen_entities: set[str] = set()
    for entity, span, alias in sorted(matches, key=lambda item: (-(item[1][1] - item[1][0]), -len(item[2]), item[0].canonical.casefold())):
        key = entity.canonical.casefold()
        if key in seen_entities:
            continue
        if any(_spans_overlap(span, selected_span) for _, selected_span, _ in selected):
            continue
        selected.append((entity, span, alias))
        seen_entities.add(key)
    selected.sort(key=lambda item: item[1][0])
    return [entity for entity, _, _ in selected]


def _spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _section_titles(document: dict) -> list[str]:
    titles: list[str] = []
    for section in _list_of_dicts(document.get("sections")):
        title = str(section.get("title") or "")
        if title:
            titles.append(title)
        for heading in _list(section.get("heading_path")):
            titles.append(heading)
    return list(dict.fromkeys(titles))


def _list(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _list_of_dicts(value: object) -> list[dict]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []
