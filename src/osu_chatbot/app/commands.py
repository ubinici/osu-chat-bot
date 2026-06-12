from __future__ import annotations

from pathlib import Path

from ..config import AppConfig
from ..corpus.chunker import build_chunks
from ..corpus.parser import parse_news_file, parse_wiki_file
from ..corpus.sources import iter_news_files, iter_wiki_files
from ..domain.artifacts import (
    CHUNKS_FILE,
    DOCUMENTS_FILE,
    INGEST_REPORT_FILE,
    LINK_ALIAS_REVIEW_FILE,
    ENTITY_CANDIDATES_FILE,
    ENTITY_CANDIDATES_REPORT_FILE,
    ENTITY_NORMALIZATION_FILE,
    ENTITY_NORMALIZATION_REPORT_FILE,
    ENTITY_NORMALIZATION_REVIEW_FILE,
    STATS_REPORT_FILE,
    TERMS_FILE,
    VALIDATION_REPORT_FILE,
    load_chunks,
    write_json,
    write_jsonl,
)
from ..evaluation.runner import run_evaluation
from ..generation.answerer import answer_question
from ..indexing.pipeline import IndexOptions, build_index
from ..knowledge.links import build_link_artifacts
from ..knowledge.ner import build_entity_candidates_from_artifacts
from ..knowledge.normalization import build_entity_normalization_artifacts
from ..knowledge.terms import build_terms_from_artifacts
from ..quality.stats import build_stats_report
from ..quality.validation import build_validation_report
from ..retrieval.service import Retriever


def ingest(config: AppConfig) -> tuple[list[dict], list]:
    root = config.corpus.osu_wiki_path
    if not root.exists():
        raise FileNotFoundError(f"osu-wiki path does not exist: {root}")

    documents: list[dict] = [
        parse_wiki_file(path, root, lang=config.corpus.language)
        for path in iter_wiki_files(root, lang=config.corpus.language)
    ]
    if config.corpus.include_news:
        documents.extend(parse_news_file(path, root) for path in iter_news_files(root))

    chunks = []
    for document in documents:
        chunks.extend(build_chunks(document))

    artifact_dir = config.artifacts.path
    write_jsonl(artifact_dir / DOCUMENTS_FILE, documents)
    write_jsonl(artifact_dir / CHUNKS_FILE, chunks)
    write_json(
        artifact_dir / INGEST_REPORT_FILE,
        {
            "documents": len(documents),
            "chunks": len(chunks),
            "wiki_documents": sum(1 for doc in documents if doc.get("source") == "osu-wiki"),
            "news_documents": sum(1 for doc in documents if doc.get("source") == "osu-news"),
            "sections": sum(int(doc.get("section_count", 0)) for doc in documents),
            "links": sum(int(doc.get("edge_count", 0)) for doc in documents),
            "citations": sum(int(doc.get("citation_count", 0)) for doc in documents),
            "tables": sum(int(doc.get("table_count", 0)) for doc in documents),
            "formulas": sum(int(doc.get("formula_count", 0)) for doc in documents),
            "images": sum(int(doc.get("image_count", 0)) for doc in documents),
            "language": config.corpus.language,
            "osu_wiki_path": str(root),
            "document_artifact": DOCUMENTS_FILE,
            "chunk_artifact": CHUNKS_FILE,
        },
    )
    return documents, chunks


def run_ingest(config: AppConfig) -> int:
    documents, chunks = ingest(config)
    print(f"Ingested {len(documents)} documents into {len(chunks)} chunks.")
    print(f"Artifacts: {config.artifacts.path}")
    return 0


def run_terms(config: AppConfig) -> int:
    entities = build_terms_from_artifacts(config.artifacts.path)
    print(f"Built {len(entities)} entities.")
    print(f"Artifacts: {config.artifacts.path / TERMS_FILE}")
    return 0


def run_links(config: AppConfig) -> int:
    report = build_link_artifacts(config.artifacts.path)
    print(f"Raw links: {report['raw_links']}")
    print(f"Internal wiki links: {report['internal_wiki_links']}")
    print(
        "Candidates: "
        f"accepted={report['accepted_candidates']}, "
        f"review={report['review_candidates']}, "
        f"rejected={report['rejected_candidates']}"
    )
    print(f"Artifacts: {config.artifacts.path / LINK_ALIAS_REVIEW_FILE}")
    return 0


def run_entities(
    config: AppConfig,
    *,
    backend: str,
    model_name: str | None,
    threshold: float,
    limit: int | None,
    sampling: str,
    labels: list[str] | None,
    label_profile: str,
    scoped_labels: bool,
    max_text_chars: int,
) -> int:
    input_artifact_dir = artifact_read_path(config)
    output_artifact_dir = config.artifacts.path
    report = build_entity_candidates_from_artifacts(
        input_artifact_dir,
        output_dir=output_artifact_dir,
        backend=backend,
        model_name=model_name,
        threshold=threshold,
        limit=limit,
        sampling=sampling,
        labels=labels,
        label_profile=label_profile,
        scoped_labels=scoped_labels,
        max_text_chars=max_text_chars,
    )
    print(f"Backend: {report['backend']} ({report['source_model']})")
    print(f"Chunks scanned: {report['chunks_scanned']}")
    print(f"Documents scanned: {report['documents_scanned']}")
    print(f"Sampling: {report['sampling']}")
    print(f"Candidates: {report['candidates']}")
    print(f"Unique entity/label pairs: {report['unique_entity_label_pairs']}")
    print(f"Label profile: {report['label_profile']} (scoped={report['scoped_labels']})")
    print(f"Candidates artifact: {output_artifact_dir / ENTITY_CANDIDATES_FILE}")
    print(f"Report: {output_artifact_dir / ENTITY_CANDIDATES_REPORT_FILE}")
    return 0


def run_normalize_entities(config: AppConfig) -> int:
    input_artifact_dir = artifact_read_path(config)
    output_artifact_dir = config.artifacts.path
    report = build_entity_normalization_artifacts(
        output_artifact_dir,
        source_artifact_dir=input_artifact_dir,
        output_dir=output_artifact_dir,
    )
    print(f"Candidates: {report['candidates']}")
    print(f"Normalized rows: {report['normalized_rows']}")
    print(f"Linked rows: {report['linked_rows']}")
    print(f"Decisions: {_format_counts(report['by_decision'])}")
    print(f"Normalized artifact: {config.artifacts.path / ENTITY_NORMALIZATION_FILE}")
    print(f"Review CSV: {config.artifacts.path / ENTITY_NORMALIZATION_REVIEW_FILE}")
    print(f"Report: {config.artifacts.path / ENTITY_NORMALIZATION_REPORT_FILE}")
    return 0


def run_stats(config: AppConfig) -> int:
    report = build_stats_report(config.artifacts.path)
    documents = report["documents"]
    chunks = report["chunks"]
    structure = report["structure"]
    terms = report["terms"]
    print(f"Documents: {documents['total']}")
    print(f"  by source: {_format_counts(documents['by_source'])}")
    print(f"Chunks: {chunks['total']}")
    print(f"  by type: {_format_counts(chunks['by_type'])}")
    print(
        "Structure: "
        f"sections={structure['sections']}, links={structure['links']}, citations={structure['citations']}, "
        f"tables={structure['tables']}, formulas={structure['formulas']}, images={structure['images']}"
    )
    print(f"Terms: {terms['total_entities']} entities, {terms['total_aliases']} aliases")
    print(f"Report: {config.artifacts.path / STATS_REPORT_FILE}")
    return 0


def run_validate(config: AppConfig) -> int:
    report = build_validation_report(config.artifacts.path)
    summary = report["summary"]
    print(f"Issues: {summary['issues']}")
    print(f"  by severity: {_format_counts(summary['by_severity'])}")
    print(f"  top kinds: {_format_counts(summary['by_kind'], limit=10)}")
    print(f"Report: {config.artifacts.path / VALIDATION_REPORT_FILE}")
    return 1 if summary["by_severity"].get("error", 0) else 0


def run_index(config: AppConfig, *, batch_size: int, offset: int, limit: int | None, resume: bool, quiet: bool) -> int:
    report = build_index(
        config,
        IndexOptions(batch_size=batch_size, offset=offset, limit=limit, resume=resume),
        progress=None if quiet else print_index_progress,
    )
    status = "complete" if report["complete"] else f"next offset {report['next_offset']}"
    print(f"Indexed {report['chunks_indexed']} chunks into Qdrant collection `{report['collection']}` ({status}).")
    return 0


def run_inspect(config: AppConfig, question: str, *, keyword_only: bool = False) -> int:
    ensure_query_artifacts(config)
    retriever = Retriever(config, use_dense=not keyword_only)
    entities, results = retriever.search(question)
    print_entities(entities)
    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        heading = " > ".join(chunk.heading_path) if chunk.heading_path else chunk.title
        chunk_type = chunk.metadata.get("chunk_type", "chunk")
        source = chunk.metadata.get("source", chunk.source_type)
        print(
            f"\n[{rank}] score={result.score:.3f} "
            f"dense={result.dense_score:.3f} "
            f"keyword={result.keyword_score:.3f} "
            f"entity={result.entity_score:.3f} "
            f"doc={result.document_score:.3f}"
        )
        print(f"{chunk.title} | {heading} | {source}/{chunk_type}")
        print(chunk.osu_url)
        print(_preview(chunk.text))
    return 0


def run_query(config: AppConfig, question: str, *, keyword_only: bool = False) -> int:
    ensure_query_artifacts(config)
    retriever = Retriever(config, use_dense=not keyword_only)
    entities, results = retriever.search(question)
    print_entities(entities)
    answer = answer_question(question, results, config.ollama)
    print("\n" + answer)
    print("\nSources:")
    for index, result in enumerate(results, start=1):
        print(f"[{index}] {result.chunk.title} - {result.chunk.osu_url}")
    return 0


def run_eval(config: AppConfig, dataset: Path, *, use_dense: bool = False, output: Path | None = None) -> int:
    report = run_evaluation(config, dataset, use_dense=use_dense)
    summary = report["summary"]
    print(f"Examples: {summary['examples']}")
    print(f"Judged: {summary['judged_examples']}")
    print(f"Matches: {summary['matches']}")
    print(f"Retrieval accuracy: {summary['retrieval_accuracy']:.3f}")
    for category, category_summary in report.get("by_category", {}).items():
        print(
            f"  {category}: "
            f"{category_summary['matches']}/{category_summary['judged_examples']} "
            f"({category_summary['retrieval_accuracy']:.3f})"
        )
    if output:
        write_json(output, report)
        print(f"Report: {output}")
    return 0


def print_index_progress(event: dict[str, object]) -> None:
    if event["event"] == "start":
        start_offset = int(event["start_offset"])
        end_offset = int(event["end_offset"])
        print(
            "Indexing "
            f"{event['chunks_to_index']} chunks "
            f"({start_offset}..{end_offset - 1}) "
            f"of {event['total_chunks']} with batch_size={event['batch_size']}"
        )
        print(f"Embedding model: {event['embedding_model']}")
        print(f"Collection: {event['collection']}")
        return
    if event["event"] == "load_model_start":
        print(f"Loading embedding model: {event['embedding_model']}", flush=True)
        return
    if event["event"] == "load_model_finish":
        print(f"Embedding model loaded in {float(event['seconds']):.2f}s", flush=True)
        return
    if event["event"] == "qdrant_start":
        print(f"Connecting Qdrant: {event['qdrant_url']} / {event['collection']}", flush=True)
        return
    if event["event"] == "qdrant_finish":
        print("Qdrant collection ready", flush=True)
        return
    if event["event"] == "batch":
        indexed = int(event["indexed_this_run"])
        target = int(event["chunks_to_index"])
        percent = 100 * indexed / target if target else 100
        cps = float(event["chunks_per_second"])
        remaining = max(0, target - indexed)
        eta_seconds = remaining / cps if cps else 0
        print(
            f"[{percent:6.2f}%] "
            f"chunks {event['batch_start']}..{int(event['batch_end']) - 1} "
            f"batch={event['batch_size']} "
            f"embed={float(event['embed_seconds']):.2f}s "
            f"upsert={float(event['upsert_seconds']):.2f}s "
            f"rate={cps:.2f}/s "
            f"eta={format_duration(eta_seconds)} "
            f"next_offset={event['next_offset']}",
            flush=True,
        )
        return
    if event["event"] == "finish":
        print(f"Index report: {event.get('end_offset')} / {event.get('total_chunks')} chunks processed")


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def ensure_query_artifacts(config: AppConfig) -> None:
    artifact_dir = artifact_read_path(config)
    if not load_chunks(artifact_dir / CHUNKS_FILE):
        raise RuntimeError("No chunks found. Run `osu-bot ingest` first.")


def artifact_read_path(config: AppConfig) -> Path:
    return config.artifacts.source_path or config.artifacts.path


def print_entities(entities) -> None:
    if not entities:
        print("Entities: none detected")
        return
    print("Entities: " + ", ".join(entity.canonical for entity in entities[:12]))


def _preview(text: str, limit: int = 500) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _format_counts(counts: dict, limit: int | None = None) -> str:
    items = list(counts.items())
    if limit is not None:
        items = items[:limit]
    return ", ".join(f"{key}={value}" for key, value in items) if items else "none"
