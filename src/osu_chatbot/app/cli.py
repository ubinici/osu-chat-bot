from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..config import load_config
from ..knowledge.ner import DEFAULT_LABEL_PROFILE
from ..knowledge.label_profiles import LABEL_PROFILES
from . import commands


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="osu-bot", description="Small cited osu! RAG chatbot")
    parser.add_argument("--config", default="config.toml", help="Path to TOML config file")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ingest", help="Parse osu-wiki markdown into documents and chunks")
    subparsers.add_parser("terms", help="Build osu!-specific terminology dictionary")
    subparsers.add_parser("links", help="Build reviewable hyperlink alias artifacts")
    subparsers.add_parser("aliases", help="Build the runtime document alias artifact")
    entities_parser = subparsers.add_parser("entities", help="Extract generative entity candidates from chunks")
    entities_parser.add_argument("--backend", default="gliner", choices=["gliner"], help="Entity extraction backend")
    entities_parser.add_argument("--model", default=None, help="Backend model name")
    entities_parser.add_argument("--threshold", type=float, default=0.5, help="Minimum backend confidence score")
    entities_parser.add_argument("--limit", type=int, default=None, help="Maximum number of chunks to scan")
    entities_parser.add_argument(
        "--sampling",
        default="balanced",
        choices=["balanced", "sequential"],
        help="How to choose chunks when --limit is set",
    )
    entities_parser.add_argument(
        "--label-profile",
        default=DEFAULT_LABEL_PROFILE,
        choices=sorted(LABEL_PROFILES),
        help="Built-in label profile to use when --label is not provided",
    )
    entities_parser.add_argument("--label", action="append", dest="labels", help="Entity label to extract; may be repeated")
    entities_parser.add_argument(
        "--no-scoped-labels",
        action="store_true",
        help="Do not narrow Main Page profile labels based on chunk document path",
    )
    entities_parser.add_argument("--max-text-chars", type=int, default=3500, help="Maximum characters per chunk sent to the model")
    subparsers.add_parser("normalize-entities", help="Normalize generative entity candidates into reviewable canonical groups")
    subparsers.add_parser("stats", help="Summarize structured database artifacts")
    subparsers.add_parser("validate", help="Validate structured database artifacts")

    index_parser = subparsers.add_parser("index", help="Embed chunks and persist the Qdrant collection")
    index_parser.add_argument("--batch-size", type=int, default=64, help="Chunks to embed/upsert per batch")
    index_parser.add_argument("--offset", type=int, default=0, help="Chunk offset to start from")
    index_parser.add_argument("--limit", type=int, default=None, help="Maximum chunks to index in this run")
    index_parser.add_argument("--resume", action="store_true", help="Resume from artifacts/rag/index_state.json")
    index_parser.add_argument("--quiet", action="store_true", help="Only print final index result")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect retrieval results without calling the LLM")
    inspect_parser.add_argument("question")

    query_parser = subparsers.add_parser("query", help="Ask a cited RAG question")
    query_parser.add_argument("question")

    serve_parser = subparsers.add_parser("serve", help="Run the HTTP chat API")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)

    eval_parser = subparsers.add_parser("eval", help="Run retrieval evaluation from a JSONL dataset")
    eval_parser.add_argument("dataset", type=Path)
    eval_parser.add_argument("--output", type=Path, help="Optional JSON report path")
    eval_parser.add_argument("--top-k", type=int, help="Override retrieval depth for this evaluation")

    topic_dataset_parser = subparsers.add_parser(
        "build-topic-dataset",
        help="Promote reviewed feedback into a query-to-topic JSONL dataset",
    )
    topic_dataset_parser.add_argument("feedback", type=Path, help="Raw feedback events JSONL")
    topic_dataset_parser.add_argument("reviews", type=Path, help="Offline feedback reviews JSONL")
    topic_dataset_parser.add_argument("--output", type=Path, required=True, help="Output query-topic JSONL")
    topic_dataset_parser.add_argument(
        "--held-out",
        type=Path,
        help="Evaluation JSONL whose exact normalized queries must not be promoted",
    )

    topic_model_parser = subparsers.add_parser(
        "eval-topic-model",
        help="Compare semantic query-topic kNN with the deterministic alias resolver",
    )
    topic_model_parser.add_argument("dataset", type=Path, help="Query-topic JSONL dataset")
    topic_model_parser.add_argument("--alias-artifact", type=Path, help="Alias JSONL override")
    topic_model_parser.add_argument("--split", default="validation", choices=["validation", "test"])
    topic_model_parser.add_argument("--top-k", type=int, default=3)
    topic_model_parser.add_argument("--output", type=Path, help="Optional JSON report path")

    args = parser.parse_args(argv)
    config = load_config(args.config)

    try:
        if args.command == "ingest":
            return commands.run_ingest(config)
        if args.command == "terms":
            return commands.run_terms(config)
        if args.command == "links":
            return commands.run_links(config)
        if args.command == "aliases":
            return commands.run_aliases(config)
        if args.command == "entities":
            return commands.run_entities(
                config,
                backend=args.backend,
                model_name=args.model,
                threshold=args.threshold,
                limit=args.limit,
                sampling=args.sampling,
                labels=args.labels,
                label_profile=args.label_profile,
                scoped_labels=not args.no_scoped_labels,
                max_text_chars=args.max_text_chars,
            )
        if args.command == "normalize-entities":
            return commands.run_normalize_entities(config)
        if args.command == "stats":
            return commands.run_stats(config)
        if args.command == "validate":
            return commands.run_validate(config)
        if args.command == "index":
            return commands.run_index(
                config,
                batch_size=args.batch_size,
                offset=args.offset,
                limit=args.limit,
                resume=args.resume,
                quiet=args.quiet,
            )
        if args.command == "inspect":
            return commands.run_inspect(config, args.question)
        if args.command == "query":
            return commands.run_query(config, args.question)
        if args.command == "serve":
            return commands.run_server(config, host=args.host, port=args.port)
        if args.command == "eval":
            return commands.run_eval(config, args.dataset, output=args.output, top_k=args.top_k)
        if args.command == "build-topic-dataset":
            return commands.run_build_topic_dataset(
                args.feedback,
                args.reviews,
                output=args.output,
                held_out=args.held_out,
            )
        if args.command == "eval-topic-model":
            return commands.run_eval_topic_model(
                config,
                args.dataset,
                alias_artifact=args.alias_artifact,
                split=args.split,
                top_k=args.top_k,
                output=args.output,
            )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
