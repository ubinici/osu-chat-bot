from __future__ import annotations

from collections import Counter
from collections import defaultdict
import math
import re

from ..config import AppConfig
from ..domain.artifacts import CHUNKS_FILE, DOCUMENTS_FILE, TERMS_FILE, load_chunks, load_entities, load_records
from ..domain.models import Entity, SearchResult
from .intent import QueryIntent, classify_query, normalize_token
from ..knowledge.terms import detect_entities
from .dense import DenseRetriever
from .lexical import document_id, section_text_parts, tokenize
from .ranker import result_sort_key

PROCEDURAL_HINT_RE = re.compile(r"(?m)(^\s*\d+\.|\b(open|click|download|install|use)\b)")


class Retriever:
    def __init__(self, config: AppConfig, use_dense: bool = True):
        self.config = config
        artifact_dir = config.artifacts.source_path or config.artifacts.path
        self.chunks = {chunk.id: chunk for chunk in load_chunks(artifact_dir / CHUNKS_FILE)}
        self.documents = {document_id(doc): doc for doc in load_records(artifact_dir / DOCUMENTS_FILE) if document_id(doc)}
        self.entities = load_entities(artifact_dir / TERMS_FILE)
        self._token_counts: dict[str, Counter[str]] = {}
        self._document_token_counts: dict[str, Counter[str]] = {}
        self._title_tokens: dict[str, set[str]] = {}
        self._search_text: dict[str, str] = {}
        self._metadata_text: dict[str, str] = {}
        self._postings: dict[str, set[str]] = defaultdict(set)
        self._chunks_by_document: dict[str, set[str]] = defaultdict(set)
        self._build_keyword_index()
        self.use_dense = use_dense
        self._dense = DenseRetriever(config, self.chunks, enabled=use_dense)

    def search(self, query: str, final_top_k: int | None = None) -> tuple[list[Entity], list[SearchResult]]:
        final_top_k = final_top_k or self.config.retrieval.final_top_k
        intent = classify_query(query)
        detected = detect_entities(query, self.entities)
        document_scores = self._document_scores(query, detected, intent)
        dense_scores = self._dense_scores(query)
        candidates = set(dense_scores)
        candidates.update(self._document_candidate_ids(document_scores))
        candidates.update(self._keyword_candidate_ids(query))
        candidates.update(self._entity_candidate_ids(detected))
        if not candidates:
            candidates.update(list(self.chunks)[: self.config.retrieval.dense_top_k])
        keyword_scores = self._keyword_scores(query, candidates, intent)
        entity_scores = self._entity_scores(detected, candidates)
        chunk_document_scores = self._chunk_document_scores(document_scores, candidates)

        results = [
            SearchResult(
                chunk=self.chunks[chunk_id],
                dense_score=dense_scores.get(chunk_id, 0.0),
                keyword_score=keyword_scores.get(chunk_id, 0.0),
                entity_score=entity_scores.get(chunk_id, 0.0),
                document_score=chunk_document_scores.get(chunk_id, 0.0),
            )
            for chunk_id in candidates
            if chunk_id in self.chunks
        ]
        results.sort(key=result_sort_key)
        return detected, results[:final_top_k]

    def _document_scores(self, query: str, detected: list[Entity], intent: QueryIntent) -> dict[str, float]:
        if not self.documents:
            return {}
        query_terms = Counter([*tokenize(query), *intent.expanded_terms])
        if not query_terms and not detected:
            return {}
        clean_query = " ".join(query_terms)
        entity_names = {
            name
            for entity in detected
            for name in [entity.canonical.casefold(), *(alias.casefold() for alias in entity.aliases)]
        }
        scores: dict[str, float] = {}
        for doc_id, document in self.documents.items():
            doc_terms = self._document_terms_for(doc_id)
            overlap = sum(min(count, doc_terms.get(term, 0)) for term, count in query_terms.items())
            score = math.log1p(overlap) * 1.8 if overlap else 0.0
            title_key = str(document.get("title") or "").casefold()
            page_key = doc_id.replace("_", " ").replace("/", " ").casefold()
            tag_keys = {str(tag).casefold() for tag in document.get("tags", []) if isinstance(document.get("tags"), list)}
            if clean_query and clean_query == title_key:
                score += 10.0
            elif clean_query and (clean_query in title_key or clean_query in page_key):
                score += 4.0
            if entity_names and (title_key in entity_names or page_key in entity_names or any(tag in entity_names for tag in tag_keys)):
                score += 8.0
            tag_overlap = sum(1 for term in query_terms if term in tag_keys)
            score += tag_overlap * 2.5
            title_terms = set(tokenize(str(document.get("title") or "")))
            page_tail_terms = set(tokenize(doc_id.rsplit("/", 1)[-1].replace("_", " ")))
            score += sum(1 for term in query_terms if term in title_terms) * 3.0
            score += sum(1 for term in query_terms if term in page_tail_terms) * 2.0
            score += intent.document_hints.get(doc_id, 0.0)
            if intent.has("troubleshooting") and str(document.get("subculture") or "") == "help":
                score += 1.5
            if score:
                scores[doc_id] = score
        return scores

    def _document_candidate_ids(self, document_scores: dict[str, float], limit: int = 12) -> set[str]:
        candidates: set[str] = set()
        for doc_id, _ in sorted(document_scores.items(), key=lambda item: item[1], reverse=True)[:limit]:
            candidates.update(self._chunks_by_document.get(doc_id, set()))
            prefix = f"{doc_id}/"
            for child_doc_id, chunk_ids in self._chunks_by_document.items():
                if child_doc_id.startswith(prefix):
                    candidates.update(chunk_ids)
        return candidates

    def _chunk_document_scores(self, document_scores: dict[str, float], candidate_ids: set[str]) -> dict[str, float]:
        scores: dict[str, float] = {}
        if not document_scores:
            return scores
        for chunk_id in candidate_ids:
            doc_id = self.chunks[chunk_id].document_id
            score = document_scores.get(doc_id, 0.0)
            if not score:
                parent_scores = [value * 0.7 for parent_id, value in document_scores.items() if doc_id.startswith(f"{parent_id}/")]
                score = max(parent_scores, default=0.0)
            if score:
                chunk_type = str(self.chunks[chunk_id].metadata.get("chunk_type", ""))
                type_multiplier = 1.15 if chunk_type == "article" else 1.0
                scores[chunk_id] = score * type_multiplier
        return scores

    def _dense_scores(self, query: str) -> dict[str, float]:
        return self._dense.scores(query)

    def _keyword_scores(self, query: str, candidate_ids: set[str], intent: QueryIntent) -> dict[str, float]:
        query_terms = Counter([*tokenize(query), *intent.expanded_terms])
        if not query_terms:
            return {}
        clean_query = " ".join(query_terms)
        scores: dict[str, float] = {}
        for chunk_id in candidate_ids:
            chunk = self.chunks[chunk_id]
            terms = self._token_counts_for(chunk_id)
            overlap = sum(min(count, terms.get(term, 0)) for term, count in query_terms.items())
            if overlap:
                title_bonus = sum(1 for term in query_terms if term in self._title_tokens.get(chunk_id, set())) * 0.25
                exact_bonus = 0.0
                title_key = chunk.title.casefold()
                heading_key = chunk.heading_path[-1].casefold() if chunk.heading_path else ""
                if title_key == clean_query:
                    exact_bonus += 8.0
                elif clean_query and clean_query in title_key:
                    exact_bonus += 1.0
                if heading_key == clean_query:
                    exact_bonus += 4.0
                if chunk.source_type == "wiki":
                    exact_bonus += 0.15
                if intent.has("access") and PROCEDURAL_HINT_RE.search(chunk.text.casefold()):
                    exact_bonus += 1.25
                heading_tokens = set(tokenize(" ".join(chunk.heading_path)))
                exact_bonus += sum(1 for term in query_terms if term in heading_tokens) * 0.9
                if intent.has("troubleshooting") and chunk.document_id.startswith("Help_centre"):
                    exact_bonus += 0.5
                if intent.has("performance") and chunk.document_id == "Performance_troubleshooting":
                    exact_bonus += 1.5
                scores[chunk_id] = math.log1p(overlap) + title_bonus + exact_bonus
        return scores

    def _entity_scores(self, detected: list[Entity], candidate_ids: set[str]) -> dict[str, float]:
        if not detected:
            return {}
        scores: dict[str, float] = {}
        for chunk_id in candidate_ids:
            chunk = self.chunks[chunk_id]
            haystack = self._search_text_for(chunk_id)
            score = 0.0
            for entity in detected:
                if any(alias.casefold() in haystack for alias in entity.aliases):
                    score += entity.score
                entity_names = {entity.canonical.casefold(), *(alias.casefold() for alias in entity.aliases)}
                title_key = chunk.title.casefold()
                heading_key = chunk.heading_path[-1].casefold() if chunk.heading_path else ""
                if title_key in entity_names:
                    score += 8.0
                if heading_key in entity_names:
                    score += 3.0
            if score:
                scores[chunk_id] = score
        return scores

    def _keyword_candidate_ids(self, query: str) -> set[str]:
        candidates: set[str] = set()
        for term in tokenize(query):
            candidates.update(self._postings.get(term, set()))
        return candidates

    def _entity_candidate_ids(self, detected: list[Entity]) -> set[str]:
        candidates: set[str] = set()
        for entity in detected:
            for alias in entity.aliases:
                candidates.update(self._postings.get(alias.casefold(), set()))
                for token in tokenize(alias):
                    candidates.update(self._postings.get(token, set()))
        return candidates

    def _build_keyword_index(self) -> None:
        for chunk_id, chunk in self.chunks.items():
            self._chunks_by_document[chunk.document_id].add(chunk_id)
            metadata_text = " ".join(
                [
                    chunk.title,
                    " ".join(chunk.heading_path),
                    " ".join(chunk.tags),
                    chunk.file_path,
                    str(chunk.metadata.get("chunk_type", "")),
                    str(chunk.metadata.get("domain", "")),
                    str(chunk.metadata.get("subculture", "")),
                    str(chunk.metadata.get("series_primary", "")),
                ]
            )
            self._metadata_text[chunk_id] = metadata_text.casefold()
            tokens = set(tokenize(metadata_text))
            self._title_tokens[chunk_id] = set(tokenize(chunk.title))
            for token in tokens:
                self._postings[token].add(chunk_id)
            for exact in [chunk.title, *chunk.heading_path, *chunk.tags, str(chunk.metadata.get("chunk_type", ""))]:
                exact_key = exact.casefold()
                if exact_key:
                    self._postings[exact_key].add(chunk_id)

    def _token_counts_for(self, chunk_id: str) -> Counter[str]:
        if chunk_id not in self._token_counts:
            chunk = self.chunks[chunk_id]
            search_text = " ".join(
                [
                    chunk.title,
                    " ".join(chunk.heading_path),
                    " ".join(chunk.tags),
                    str(chunk.metadata.get("chunk_type", "")),
                    str(chunk.metadata.get("domain", "")),
                    str(chunk.metadata.get("subculture", "")),
                    chunk.text,
                ]
            )
            self._token_counts[chunk_id] = Counter(tokenize(search_text))
        return self._token_counts[chunk_id]

    def _search_text_for(self, chunk_id: str) -> str:
        if chunk_id not in self._search_text:
            chunk = self.chunks[chunk_id]
            self._search_text[chunk_id] = " ".join(
                [
                    chunk.title,
                    " ".join(chunk.heading_path),
                    " ".join(chunk.tags),
                    str(chunk.metadata.get("chunk_type", "")),
                    str(chunk.metadata.get("domain", "")),
                    str(chunk.metadata.get("subculture", "")),
                    chunk.text,
                ]
            ).casefold()
        return self._search_text[chunk_id]

    def _document_terms_for(self, doc_id: str) -> Counter[str]:
        if doc_id not in self._document_token_counts:
            document = self.documents[doc_id]
            text = " ".join(
                [
                    str(document.get("title") or ""),
                    doc_id.replace("_", " ").replace("/", " "),
                    " ".join(str(tag) for tag in document.get("tags", []) if isinstance(document.get("tags"), list)),
                    " ".join(str(tag) for tag in document.get("series_tags", []) if isinstance(document.get("series_tags"), list)),
                    str(document.get("domain") or ""),
                    str(document.get("subculture") or ""),
                    str(document.get("topic") or ""),
                    str(document.get("slug") or "").replace("-", " "),
                    " ".join(section_text_parts(document)),
                ]
            )
            self._document_token_counts[doc_id] = Counter(tokenize(text))
        return self._document_token_counts[doc_id]
