from __future__ import annotations

from collections import Counter
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import re

from ..domain.artifacts import (
    CHUNKS_FILE,
    ENTITY_CANDIDATES_FILE,
    ENTITY_CANDIDATES_REPORT_FILE,
    load_chunks,
    write_json,
    write_jsonl,
)
from ..domain.models import Chunk
from .label_profiles import MAIN_PAGE_LABEL_PROFILE, labels_for_chunk, labels_for_profile

DEFAULT_LABEL_PROFILE = MAIN_PAGE_LABEL_PROFILE
GENERIC_CANDIDATES = {
    "article",
    "article naming",
    "articles that need help",
    "code",
    "comments",
    "disambiguation",
    "disambiguation article",
    "disambiguation articles",
    "end of line sequence",
    "formatting",
    "front matter",
    "hatnote",
    "hatnotes",
    "headings",
    "html",
    "html comments",
    "keywords and commands",
    "line breaks",
    "lists",
    "notice",
    "outdated articles",
    "outdated translations",
    "paragraphs",
    "stacked hatnotes and notices",
    "topic",
    "translations without reviews",
    "warning",
}
GENERIC_SUFFIXES = (
    " article",
    " articles",
    " heading",
    " headings",
    " paragraph",
    " paragraphs",
)


class EntityExtractor(Protocol):
    backend: str
    source_model: str

    def extract(self, text: str, labels: list[str]) -> list[dict]:
        ...


@dataclass(frozen=True)
class EntityCandidate:
    text: str
    label: str
    canonical_guess: str
    score: float
    chunk_id: str
    document_id: str
    source_type: str
    title: str
    heading_path: list[str]
    start: int | None
    end: int | None
    context: str
    backend: str
    source_model: str


def build_entity_candidates_from_artifacts(
    artifact_dir: Path,
    *,
    output_dir: Path | None = None,
    backend: str = "gliner",
    labels: list[str] | None = None,
    label_profile: str = DEFAULT_LABEL_PROFILE,
    scoped_labels: bool = True,
    model_name: str | None = None,
    threshold: float = 0.5,
    limit: int | None = None,
    sampling: str = "balanced",
    max_text_chars: int = 3500,
) -> dict[str, object]:
    output_dir = output_dir or artifact_dir
    chunks = load_chunks(artifact_dir / CHUNKS_FILE)
    selected_chunks = select_chunks(chunks, limit=limit, sampling=sampling)
    extractor = create_extractor(backend, model_name=model_name, threshold=threshold)
    active_labels = labels or labels_for_profile(label_profile)
    active_scoped_labels = scoped_labels and labels is None
    candidates = extract_entity_candidates(
        selected_chunks,
        extractor,
        labels=active_labels,
        label_profile=label_profile,
        scoped_labels=active_scoped_labels,
        max_text_chars=max_text_chars,
    )
    write_jsonl(output_dir / ENTITY_CANDIDATES_FILE, candidates)

    by_label = Counter(candidate.label for candidate in candidates)
    unique_entities = {(candidate.canonical_guess.casefold(), candidate.label) for candidate in candidates}
    report = {
        "backend": backend,
        "source_model": extractor.source_model,
        "label_profile": "custom" if labels else label_profile,
        "scoped_labels": active_scoped_labels,
        "labels": active_labels,
        "threshold": threshold,
        "sampling": sampling,
        "chunks_available": len(chunks),
        "chunks_scanned": len(selected_chunks),
        "documents_scanned": len({chunk.document_id for chunk in selected_chunks}),
        "candidates": len(candidates),
        "unique_entity_label_pairs": len(unique_entities),
        "by_label": dict(by_label.most_common()),
        "candidate_artifact": ENTITY_CANDIDATES_FILE,
    }
    write_json(output_dir / ENTITY_CANDIDATES_REPORT_FILE, report)
    return report


def create_extractor(backend: str, *, model_name: str | None, threshold: float) -> EntityExtractor:
    if backend == "gliner":
        from .gliner_ner import GlinerEntityExtractor

        return GlinerEntityExtractor(model_name=model_name, threshold=threshold)
    raise ValueError(f"Unsupported entity extraction backend: {backend}")


def extract_entity_candidates(
    chunks: list[Chunk],
    extractor: EntityExtractor,
    *,
    labels: list[str],
    label_profile: str = DEFAULT_LABEL_PROFILE,
    scoped_labels: bool = False,
    max_text_chars: int = 3500,
) -> list[EntityCandidate]:
    candidates: list[EntityCandidate] = []
    for chunk in chunks:
        text = chunk.text[:max_text_chars]
        chunk_labels = labels_for_chunk(chunk, labels, profile=label_profile, scoped=scoped_labels)
        for raw in extractor.extract(text, chunk_labels):
            candidate = candidate_from_raw(raw, chunk, text, extractor)
            if candidate is not None:
                candidates.append(candidate)
    return sorted(dedupe_candidates(candidates), key=lambda item: (-item.score, item.label, item.canonical_guess.casefold()))


def select_chunks(chunks: list[Chunk], *, limit: int | None, sampling: str = "balanced") -> list[Chunk]:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be greater than zero when provided.")
    if limit is None:
        return list(chunks)
    if sampling == "sequential":
        return list(chunks[:limit])
    if sampling != "balanced":
        raise ValueError("sampling must be `balanced` or `sequential`.")

    chunks_by_document: dict[str, list[Chunk]] = defaultdict(list)
    document_order: list[str] = []
    for chunk in chunks:
        if chunk.document_id not in chunks_by_document:
            document_order.append(chunk.document_id)
        chunks_by_document[chunk.document_id].append(chunk)

    selected: list[Chunk] = []
    offset = 0
    while len(selected) < limit:
        added = False
        for document_id in document_order:
            document_chunks = chunks_by_document[document_id]
            if offset < len(document_chunks):
                selected.append(document_chunks[offset])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        offset += 1
    return selected


def candidate_from_raw(
    raw: dict,
    chunk: Chunk,
    chunk_text: str,
    extractor: EntityExtractor,
) -> EntityCandidate | None:
    text = normalize_candidate_text(str(raw.get("text") or ""))
    label = str(raw.get("label") or "").strip()
    if not text or not label or not is_useful_candidate(text):
        return None
    start = as_optional_int(raw.get("start"))
    end = as_optional_int(raw.get("end"))
    return EntityCandidate(
        text=text,
        label=label,
        canonical_guess=canonical_guess(text),
        score=float(raw.get("score") or 0.0),
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        source_type=chunk.source_type,
        title=chunk.title,
        heading_path=chunk.heading_path,
        start=start,
        end=end,
        context=context_window(chunk_text, start, end),
        backend=extractor.backend,
        source_model=extractor.source_model,
    )


def dedupe_candidates(candidates: list[EntityCandidate]) -> list[EntityCandidate]:
    best: dict[tuple[str, str, str], EntityCandidate] = {}
    for candidate in candidates:
        key = (candidate.canonical_guess.casefold(), candidate.label.casefold(), candidate.document_id)
        previous = best.get(key)
        if previous is None or candidate.score > previous.score:
            best[key] = candidate
    return list(best.values())


def normalize_candidate_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" \t\r\n.,;:()[]{}")


def canonical_guess(value: str) -> str:
    return normalize_candidate_text(value.replace("_", " ").replace("-", " "))


def is_useful_candidate(value: str) -> bool:
    key = value.casefold()
    if len(value) < 2 or len(value) > 100:
        return False
    if key in {"osu", "wiki", "article", "page", "game", "player", "players"}:
        return False
    if key in GENERIC_CANDIDATES:
        return False
    if key.endswith(GENERIC_SUFFIXES) and not key.startswith(("beatmap", "news", "wiki")):
        return False
    return bool(re.search(r"[A-Za-z0-9!]", value))


def context_window(text: str, start: int | None, end: int | None, *, window: int = 160) -> str:
    if start is None or end is None:
        return normalize_candidate_text(text[: window * 2])
    left = max(0, start - window)
    right = min(len(text), end + window)
    return normalize_candidate_text(text[left:right])


def as_optional_int(value: object) -> int | None:
    try:
        return int(value)
    except Exception:
        return None
