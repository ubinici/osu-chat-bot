from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import re
from typing import Any, Iterable

from ..domain.artifacts import (
    CANONICAL_TOPICS_FILE,
    CHUNKS_FILE,
    DOCUMENT_ALIASES_FILE,
    DOCUMENT_ALIASES_REPORT_FILE,
    DOCUMENTS_FILE,
    LINK_ALIAS_CANDIDATES_FILE,
    load_chunks,
    load_records,
    read_jsonl,
    write_json,
    write_jsonl,
)
from ..domain.models import Chunk
from ..domain.source_profile import profile_for_document
from ..retrieval.lexical import STOPWORDS, document_id, section_text_parts
from .links import GENERIC_ANCHORS
from .topics import (
    build_canonical_topic_rows,
    display_identifier,
    load_or_build_canonical_topic_rows,
    topic_rows_by_document,
)


ALIAS_TOKEN_RE = re.compile(r"[a-z0-9!+#.]+")
ACRONYM_RE = re.compile(r"\(([A-Za-z0-9!+#.]{2,8})\)")
MARKUP_RE = re.compile(r"[*_`]+")
GENERIC_ALIASES = {
    *GENERIC_ANCHORS,
    "external links",
    "overview",
    "preamble",
    "references",
    "see also",
    "trivia",
    "wiki",
}
ACRONYM_STOPWORDS = {*STOPWORDS, "new", "old", "ppy", "wiki"}


def build_document_alias_artifacts(artifact_dir: Path, *, output_dir: Path | None = None) -> dict[str, Any]:
    output_dir = output_dir or artifact_dir
    documents = load_records(artifact_dir / DOCUMENTS_FILE)
    chunks = load_chunks(artifact_dir / CHUNKS_FILE)
    link_alias_rows = read_jsonl(artifact_dir / LINK_ALIAS_CANDIDATES_FILE)
    topic_rows = read_jsonl(output_dir / CANONICAL_TOPICS_FILE) or load_or_build_canonical_topic_rows(artifact_dir, documents)
    rows = build_document_alias_rows(documents, chunks, link_alias_rows, topic_rows=topic_rows)
    write_jsonl(output_dir / DOCUMENT_ALIASES_FILE, rows)
    report = {
        "documents": len(documents),
        "chunks": len(chunks),
        "topics": len(topic_rows),
        "aliases": len(rows),
        "by_source": dict(Counter(str(row["source"]) for row in rows).most_common()),
        "by_trust_tier": dict(Counter(str(row["trust_tier"]) for row in rows).most_common()),
        "by_retrieval_lane": dict(Counter(str(row["retrieval_lane"]) for row in rows).most_common()),
        "artifact": DOCUMENT_ALIASES_FILE,
    }
    write_json(output_dir / DOCUMENT_ALIASES_REPORT_FILE, report)
    return report


def build_document_alias_rows(
    documents: list[dict[str, Any]],
    chunks: Iterable[Chunk],
    link_alias_rows: list[dict[str, Any]] | None = None,
    *,
    topic_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    docs = {document_id(doc): doc for doc in documents if document_id(doc)}
    chunks_by_document = group_chunks_by_document(chunks)
    topic_rows = topic_rows if topic_rows is not None else build_canonical_topic_rows(documents)
    builder = AliasBuilder(docs, chunks_by_document, topic_rows)
    for doc_id, document in docs.items():
        builder.add_document_aliases(doc_id, document)
        builder.add_chunk_aliases(doc_id)
    builder.add_topic_aliases()
    builder.add_accepted_link_aliases(link_alias_rows or [])
    return sorted(builder.rows, key=lambda row: (row["document_id"], row["alias_key"], row["source"], row["alias"]))


class AliasBuilder:
    def __init__(
        self,
        documents: dict[str, dict[str, Any]],
        chunks_by_document: dict[str, list[Chunk]],
        topic_rows: list[dict[str, Any]],
    ):
        self.documents = documents
        self.chunks_by_document = chunks_by_document
        self.topic_rows = topic_rows
        self.topics_by_document = topic_rows_by_document(topic_rows)
        self.rows: list[dict[str, Any]] = []
        self._seen: set[tuple[str, str, str, str]] = set()

    def add_document_aliases(self, doc_id: str, document: dict[str, Any]) -> None:
        title = str(document.get("title") or "")
        repo_rel_path = str(document.get("repo_rel_path") or "")
        tail = doc_id.rsplit("/", 1)[-1].replace("_", " ")

        self.add_alias(title, doc_id, "title", 1.0)
        self.add_alias(doc_id.replace("/", " ").replace("_", " "), doc_id, "page_id", 0.9)
        self.add_alias(tail, doc_id, "path_tail", 0.9)
        self.add_osu_prefixed_aliases(title, doc_id)
        self.add_osu_prefixed_aliases(tail, doc_id)
        if can_learn_generated_aliases(doc_id, document):
            self.add_title_acronym(title, doc_id)
        if repo_rel_path:
            self.add_alias(repo_rel_path.replace("/en.md", "").replace("/", " ").replace("_", " "), doc_id, "page_id", 0.75)

        for tag in document.get("tags", []) if isinstance(document.get("tags"), list) else []:
            self.add_alias(str(tag), doc_id, "tag", 0.35)
        for tag in document.get("series_tags", []) if isinstance(document.get("series_tags"), list) else []:
            self.add_alias(str(tag), doc_id, "tag", 0.35)
        for heading in section_text_parts(document):
            self.add_alias(heading, doc_id, "heading", 0.25)

    def add_title_acronym(self, title: str, doc_id: str) -> None:
        acronym = acronym_for(title)
        if acronym:
            self.add_alias(acronym, doc_id, "title_acronym", 0.9, allow_short=True)

    def add_chunk_aliases(self, doc_id: str) -> None:
        document = self.documents[doc_id]
        if not can_learn_generated_aliases(doc_id, document):
            return
        title = str(document.get("title") or "")
        if not title:
            return
        title_key = normalize_alias(title)
        for chunk in self.chunks_by_document.get(doc_id, [])[:4]:
            text = MARKUP_RE.sub("", chunk.text)
            prefix = text[:2000]
            for match in ACRONYM_RE.finditer(prefix):
                alias = match.group(1).strip()
                before = normalize_alias(prefix[max(0, match.start() - 120) : match.start()])
                if title_key and title_key in before:
                    self.add_alias(alias, doc_id, "chunk_acronym", 0.95, allow_short=True)

    def add_osu_prefixed_aliases(self, value: str, doc_id: str) -> None:
        key = normalize_alias(value)
        if not key.startswith("osu!"):
            return
        rest = key[len("osu!") :].strip()
        if len(rest) < 3:
            return
        self.add_alias(rest, doc_id, "title", 0.75)
        self.add_alias(f"osu {rest}", doc_id, "title", 0.75)

    def add_accepted_link_aliases(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            if str(row.get("decision") or "") != "accept":
                continue
            target = str(row.get("target_page_id") or "")
            if target not in self.documents:
                continue
            self.add_alias(
                str(row.get("alias") or ""),
                target,
                "accepted_link",
                float(row.get("confidence") or 1.0),
                allow_short=False,
            )

    def add_topic_aliases(self) -> None:
        for topic in self.topic_rows:
            target = str(topic.get("canonical_document_id") or "")
            if target not in self.documents:
                document_ids = [str(doc_id) for doc_id in topic.get("document_ids", []) if str(doc_id) in self.documents]
                target = document_ids[0] if document_ids else ""
            if not target:
                continue
            for alias in topic.get("aliases", []):
                self.add_alias(str(alias), target, "topic_alias", 0.85)
            for equivalent_id in topic.get("equivalent_document_ids", []):
                self.add_alias(display_identifier(str(equivalent_id)), target, "topic_identifier", 0.9)

    def add_alias(
        self,
        alias: str,
        target_document_id: str,
        source: str,
        confidence: float,
        *,
        allow_short: bool = False,
    ) -> None:
        key = normalize_alias(alias)
        if not key or key in GENERIC_ALIASES:
            return
        if source in {"title_acronym", "chunk_acronym"} and key in ACRONYM_STOPWORDS:
            return
        if not allow_short and len(key) < 3:
            return
        if source == "heading" and len(key.split()) < 2:
            return
        seen_key = (key, target_document_id, source, normalize_alias(alias))
        if seen_key in self._seen:
            return
        self._seen.add(seen_key)
        document = self.documents[target_document_id]
        profile = profile_for_document(document)
        topic = self.topics_by_document.get(target_document_id, {})
        topic_id = str(topic.get("topic_id") or target_document_id)
        canonical_document_id = str(topic.get("canonical_document_id") or target_document_id)
        topic_document_ids = [str(value) for value in topic.get("equivalent_document_ids", []) if str(value)]
        self.rows.append(
            {
                "alias": alias,
                "alias_key": key,
                "document_id": target_document_id,
                "canonical_document_id": canonical_document_id,
                "topic_id": topic_id,
                "topic_document_ids": topic_document_ids or [topic_id, canonical_document_id],
                "source": source,
                "confidence": round(confidence, 3),
                "target_source": document.get("source"),
                "retrieval_lane": profile.lane,
                "trust_tier": profile.trust_tier,
            }
        )


def group_chunks_by_document(chunks: Iterable[Chunk]) -> dict[str, list[Chunk]]:
    grouped: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.document_id].append(chunk)
    for items in grouped.values():
        items.sort(key=lambda chunk: chunk.chunk_index)
    return grouped


def can_learn_generated_aliases(doc_id: str, document: dict[str, Any]) -> bool:
    title = normalize_alias(str(document.get("title") or ""))
    return (
        str(document.get("source") or "") == "osu-wiki"
        and not doc_id.startswith("Disambiguation")
        and "disambiguation" not in title
    )


def query_alias_keys(query: str, max_ngram: int = 6) -> set[str]:
    normalized = normalize_alias(query)
    if not normalized:
        return set()
    tokens = ALIAS_TOKEN_RE.findall(normalized)
    keys = {normalized}
    for size in range(1, min(max_ngram, len(tokens)) + 1):
        for start in range(0, len(tokens) - size + 1):
            key = " ".join(tokens[start : start + size])
            if size == 1 and (key in STOPWORDS or key in GENERIC_ALIASES):
                continue
            keys.add(key)
    return keys


def normalize_alias(value: str) -> str:
    value = value.casefold()
    value = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", value)
    value = value.replace("_", " ").replace("-", " ").replace("/", " ")
    value = re.sub(r"[^a-z0-9!+#.]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def acronym_for(value: str) -> str:
    tokens = [token for token in ALIAS_TOKEN_RE.findall(normalize_alias(value)) if token not in ACRONYM_STOPWORDS]
    if len(tokens) < 2:
        return ""
    acronym = "".join(token[0] for token in tokens if token)
    if 2 <= len(acronym) <= 5:
        return acronym
    return ""
