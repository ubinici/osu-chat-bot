from __future__ import annotations

from collections import Counter, defaultdict
import math
import re

from ..config import AppConfig
from ..domain.artifacts import CHUNKS_FILE, DOCUMENTS_FILE, load_chunks, load_records
from ..domain.models import Entity, SearchResult
from ..domain.source_profile import profile_for_chunk
from .dense import DenseRetriever
from .intent import QueryIntent, classify_query
from .lexical import document_id, tokenize
from .ranker import result_sort_key
from .resolver import DocumentCandidate, DocumentResolver

PROCEDURAL_HINT_RE = re.compile(r"(?m)(^\s*\d+\.|\b(open|click|download|install|use)\b)")


class Retriever:
    def __init__(self, config: AppConfig, use_dense: bool | None = None, *, use_vectors: bool | None = None):
        self.config = config
        artifact_dir = config.artifacts.source_path or config.artifacts.path
        self.chunks = {chunk.id: chunk for chunk in load_chunks(artifact_dir / CHUNKS_FILE)}
        self.documents = {document_id(doc): doc for doc in load_records(artifact_dir / DOCUMENTS_FILE) if document_id(doc)}
        self.resolver = DocumentResolver(self.documents, artifact_dir, chunks=self.chunks.values())

        vector_fallback = config.retrieval.use_vector_fallback
        if use_dense is not None:
            vector_fallback = use_dense
        if use_vectors is not None:
            vector_fallback = use_vectors
        self.use_dense = vector_fallback
        self._dense = DenseRetriever(config, self.chunks, enabled=True) if self.use_dense else None

        self._token_counts: dict[str, Counter[str]] = {}
        self._title_tokens: dict[str, set[str]] = {}
        self._search_text: dict[str, str] = {}
        self._metadata_text: dict[str, str] = {}
        self._postings: dict[str, set[str]] = defaultdict(set)
        self._chunks_by_document: dict[str, set[str]] = defaultdict(set)
        self.last_document_candidates: list[DocumentCandidate] = []
        self._build_keyword_index()

    def search(self, query: str, final_top_k: int | None = None) -> tuple[list[Entity], list[SearchResult]]:
        final_top_k = final_top_k or self.config.retrieval.final_top_k
        intent = classify_query(query)
        candidate_limit = max(self.config.retrieval.document_top_k, self.config.retrieval.document_top_k * 4)
        resolved_candidates = self.resolver.resolve(query, intent, limit=candidate_limit)
        document_candidates = self._select_lane_candidates(resolved_candidates)
        self.last_document_candidates = document_candidates
        document_scores = {candidate.document_id: candidate.score for candidate in document_candidates}

        dense_scores = self._dense_scores(query)
        if self.use_dense and dense_scores and not document_scores:
            document_scores = self._dense_document_scores(dense_scores)

        candidates = self._document_candidate_ids(document_scores)
        if not candidates:
            candidates.update(self._keyword_candidate_ids(query))
        if self.use_dense and dense_scores and not candidates:
            candidates.update(dense_scores)
        if not candidates:
            candidates.update(list(self.chunks)[: self.config.retrieval.candidate_chunk_limit])

        keyword_scores = self._keyword_scores(query, candidates, intent)
        entity_scores = self._alias_scores(document_candidates, candidates)
        chunk_document_scores = self._chunk_document_scores(document_scores, candidates)

        results = []
        for chunk_id in candidates:
            if chunk_id not in self.chunks:
                continue
            rank_score = self._rank_score(
                chunk_id,
                dense_score=dense_scores.get(chunk_id, 0.0),
                keyword_score=keyword_scores.get(chunk_id, 0.0),
                entity_score=entity_scores.get(chunk_id, 0.0),
                document_score=chunk_document_scores.get(chunk_id, 0.0),
            )
            profile = profile_for_chunk(self.chunks[chunk_id], self.documents)
            topic_id, topic_document_ids = self.resolver.topic_for_document(self.chunks[chunk_id].document_id)
            results.append(
                SearchResult(
                    chunk=self.chunks[chunk_id],
                    dense_score=dense_scores.get(chunk_id, 0.0),
                    keyword_score=keyword_scores.get(chunk_id, 0.0),
                    entity_score=entity_scores.get(chunk_id, 0.0),
                    document_score=chunk_document_scores.get(chunk_id, 0.0),
                    rank_score=rank_score,
                    retrieval_lane=profile.lane,
                    trust_tier=profile.trust_tier,
                    source_weight=profile.weight,
                    topic_id=topic_id,
                    topic_document_ids=topic_document_ids,
                )
            )

        results.sort(key=result_sort_key)
        results = results[: self.config.retrieval.candidate_chunk_limit]
        return self._detected_entities(document_candidates), results[:final_top_k]

    def _select_lane_candidates(self, candidates: list[DocumentCandidate]) -> list[DocumentCandidate]:
        selected: list[DocumentCandidate] = []
        lane_counts: Counter[str] = Counter()
        for candidate in candidates:
            if len(selected) >= self.config.retrieval.document_top_k:
                break
            if lane_counts[candidate.retrieval_lane] >= self.config.retrieval.lane_top_k:
                continue
            selected.append(candidate)
            lane_counts[candidate.retrieval_lane] += 1
        if len(selected) < self.config.retrieval.document_top_k:
            selected_ids = {candidate.document_id for candidate in selected}
            for candidate in candidates:
                if candidate.document_id in selected_ids:
                    continue
                selected.append(candidate)
                if len(selected) >= self.config.retrieval.document_top_k:
                    break
        return selected

    def _document_candidate_ids(self, document_scores: dict[str, float]) -> set[str]:
        candidates: set[str] = set()
        for doc_id, _ in sorted(document_scores.items(), key=lambda item: item[1], reverse=True)[
            : self.config.retrieval.document_top_k
        ]:
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
            chunk = self.chunks[chunk_id]
            doc_id = chunk.document_id
            score = document_scores.get(doc_id, 0.0)
            if not score:
                parent_scores = [value * 0.7 for parent_id, value in document_scores.items() if doc_id.startswith(f"{parent_id}/")]
                score = max(parent_scores, default=0.0)
            if score:
                scores[chunk_id] = score
        return scores

    def _dense_scores(self, query: str) -> dict[str, float]:
        return self._dense.scores(query) if self._dense is not None else {}

    def _dense_document_scores(self, dense_scores: dict[str, float]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for chunk_id, score in dense_scores.items():
            chunk = self.chunks.get(chunk_id)
            if chunk is None:
                continue
            scores[chunk.document_id] = max(scores.get(chunk.document_id, 0.0), score * 10.0)
        return scores

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
                title_bonus = sum(1 for term in query_terms if term in self._title_tokens.get(chunk_id, set())) * 0.5
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
                exact_bonus += sum(1 for term in query_terms if term in heading_tokens) * 1.1
                if intent.has("troubleshooting") and chunk.document_id.startswith("Help_centre"):
                    exact_bonus += 0.5
                if intent.has("performance") and chunk.document_id == "Performance_troubleshooting":
                    exact_bonus += 1.5
                scores[chunk_id] = math.log1p(overlap) + title_bonus + exact_bonus
        return scores

    def _alias_scores(self, document_candidates: list[DocumentCandidate], candidate_ids: set[str]) -> dict[str, float]:
        aliases_by_document = {
            candidate.document_id: len(set(candidate.matched_aliases))
            for candidate in document_candidates
            if candidate.matched_aliases
        }
        if not aliases_by_document:
            return {}
        scores: dict[str, float] = {}
        for chunk_id in candidate_ids:
            chunk = self.chunks[chunk_id]
            alias_count = aliases_by_document.get(chunk.document_id, 0)
            if not alias_count:
                parent_counts = [
                    value for parent_id, value in aliases_by_document.items() if chunk.document_id.startswith(f"{parent_id}/")
                ]
                alias_count = max(parent_counts, default=0)
            if alias_count:
                scores[chunk_id] = min(8.0, 1.5 * alias_count)
        return scores

    def _rank_score(
        self,
        chunk_id: str,
        *,
        dense_score: float,
        keyword_score: float,
        entity_score: float,
        document_score: float,
    ) -> float:
        chunk = self.chunks[chunk_id]
        chunk_type = str(chunk.metadata.get("chunk_type", ""))
        type_bonus = {"article": 0.9, "section": 0.7, "table": 0.25, "formula": 0.15, "citation": 0.0}.get(chunk_type, 0.1)
        profile = profile_for_chunk(chunk, self.documents)
        base = document_score * 2.0 + keyword_score * 2.5 + entity_score * 0.7 + dense_score * 0.2 + type_bonus
        lane_bonus = {"canonical": 0.3, "troubleshooting": 0.25, "temporal": 0.05}.get(profile.lane, 0.0)
        return base * profile.weight + lane_bonus

    def _keyword_candidate_ids(self, query: str) -> set[str]:
        candidates: set[str] = set()
        for term in tokenize(query):
            candidates.update(self._postings.get(term, set()))
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

    def _detected_entities(self, document_candidates: list[DocumentCandidate]) -> list[Entity]:
        entities: list[Entity] = []
        for candidate in document_candidates[: self.config.retrieval.document_top_k]:
            if not candidate.matched_aliases:
                continue
            document = self.documents.get(candidate.document_id, {})
            entities.append(
                Entity(
                    canonical=str(document.get("title") or candidate.document_id),
                    aliases=sorted(set(candidate.matched_aliases)),
                    sources=sorted(set(candidate.reasons)),
                    score=candidate.score,
                )
            )
        return entities
