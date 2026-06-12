# osu! Chat Bot RAG Prototype

Python CLI prototype for answering osu!-related questions from the local `database/osu-wiki` checkout.

## Quick Start

```powershell
python -m pip install -e ".[dev]"
osu-bot ingest
osu-bot terms
osu-bot entities --limit 100
osu-bot normalize-entities
osu-bot stats
osu-bot validate
osu-bot index
osu-bot inspect "What is a beatmap?"
osu-bot query "What is osu!direct?"
```

## Colab Walkthrough

For a guided notebook version of the workflow, open
`notebooks/osu_chatbot_colab_walkthrough.ipynb`.

The notebook explains each step and includes runnable cells for setup, osu-wiki
checkout, artifact generation, background dense indexing, background GLiNER
entity extraction and normalization, dense seed evaluation, keyword diagnostics,
and artifact download.

Generation expects a local Ollama model, defaulting to `mistral`.

```powershell
ollama pull mistral
ollama serve
```

## Pipeline

- `ingest`: parses English wiki pages and all news posts into structured document and hierarchical chunk JSONL artifacts.
- `terms`: builds an osu!-specific entity and alias dictionary from page titles, headings, tags, paths, and link text.
- `links`: builds reviewable hyperlink alias artifacts from structured wiki/news links.
- `entities`: runs experimental zero-shot entity extraction over chunks and writes reviewable generative candidates.
- `normalize-entities`: links generative candidates to wiki pages, groups aliases, and writes reviewable normalization artifacts.
- `stats`: summarizes document/chunk/entity counts and writes `stats_report.json`.
- `validate`: checks structured artifacts for missing fields, empty content, duplicate IDs, malformed dates, large chunks, and noisy aliases.
- `index`: embeds chunks with `sentence-transformers` and persists them in Qdrant.
- `inspect`: shows detected entities and ranked retrieved chunks without calling an LLM.
- `query`: retrieves source chunks and asks Ollama to answer only from cited context.

## Package Layout

- `osu_chatbot.domain`: shared dataclasses and artifact IO.
- `osu_chatbot.corpus`: osu-wiki scanning, structured markdown/news parsing, taxonomy, and hierarchical chunk construction.
- `osu_chatbot.knowledge`: terminology and hyperlink alias extraction.
- `osu_chatbot.retrieval`: query intent, lexical scoring, dense lookup, ranking, and retrieval orchestration.
- `osu_chatbot.indexing`: embeddings and Qdrant indexing.
- `osu_chatbot.generation`: prompts, Ollama client, and answer orchestration.
- `osu_chatbot.quality`: artifact statistics and validation.
- `osu_chatbot.evaluation`: training-stage retrieval evaluation.
- `osu_chatbot.app`: command-line entrypoint and command functions.

Artifacts are written to `artifacts/rag/` by default:

- `documents_structured.jsonl`
- `chunks_hierarchical.jsonl`
- `terms.json`
- `links_raw.jsonl`
- `link_alias_candidates.jsonl`
- `link_alias_review.csv`
- `links_report.json`
- `entity_candidates_generative.jsonl`
- `entity_candidates_report.json`
- `entity_normalization_candidates.jsonl`
- `entity_normalization_review.csv`
- `entity_normalization_report.json`
- `ingest_report.json`
- `stats_report.json`
- `validation_report.json`
- `index_report.json`

## Reading Validation

`validation_report.json` is intended to separate real data-quality problems from expected corpus quirks:

- `error`: fix before indexing; these indicate broken artifacts such as duplicate IDs or empty chunk text.
- `warning`: inspect before trusting retrieval; these usually affect answer quality, such as oversized chunks.
- `info`: useful corpus notes; these are usually safe to keep unless they point at a pattern you want to normalize.

Expected wiki/news quirks such as empty layout headings, older news style differences, and very dense reference pages are mostly tracked as `info` or in `stats_report.json`.

## Generative Entity Extraction

The deterministic `terms` command is still useful as a baseline, but `entities` creates a separate candidate lane for observing zero-shot extraction quality without changing `terms.json`.

```powershell
python -m pip install -e ".[dev,entities]"
osu-bot entities --backend gliner --label-profile main-page --limit 100 --sampling balanced --threshold 0.5
```

By default this uses a Main Page-inspired label profile with categories such as game client concepts, gameplay mechanics, beatmap editor tools, ranking concepts, help/support topics, community projects, people/user groups, developer/API topics, and wiki maintenance/style topics. For clearly clustered pages, labels are scoped from the document path, so wiki style pages are evaluated as wiki-maintenance topics instead of generic client features.

Limited runs use balanced document sampling by default, so `--limit 100` scans across many articles instead of getting stuck in one alphabetically early article. Use `--sampling sequential` only when you intentionally want artifact-order scanning.

To compare against the older broad labels:

```powershell
osu-bot entities --label-profile osu-entities --no-scoped-labels --limit 100
```

To test a hand-picked label set:

```powershell
osu-bot entities --label "gameplay mechanic" --label "wiki style concept" --label "community user group"
```

The output is intended for review and comparison before promotion into the main knowledge layer.

After extraction, normalize candidates into canonical groups:

```powershell
osu-bot normalize-entities
```

This writes `entity_normalization_review.csv` for manual review. Accepted rows are linked to a wiki page when possible; review rows are plausible but need a human decision; reject rows are generic domain words or noisy candidates.

## Dense Indexing

Dense indexing can be run in short, resumable intervals:

```powershell
osu-bot index --limit 2000 --batch-size 32
osu-bot index --resume --limit 2000 --batch-size 32
```

The indexer prints each batch range, embedding time, Qdrant upsert time, throughput, ETA, and the next resume offset. It also writes `artifacts/rag/index_state.json` after every successful batch. Qdrant point IDs are deterministic, so rerunning a slice overwrites the same chunks instead of duplicating them.

If Hugging Face throttling or model downloads are a concern, pre-download the embedding model on the machine that will index, set a persistent cache, and use a token when available:

```powershell
$env:HF_HOME = "D:\hf-cache"
$env:HF_TOKEN = "<your token>"
osu-bot index --limit 100 --batch-size 16
```

After the model is cached, indexing should not need repeated Hugging Face downloads unless the cache is missing or the model changes. For cluster jobs, set the embedding config to fail fast if the model is not already cached:

```toml
[embedding]
model = "sentence-transformers/all-MiniLM-L6-v2"
cache_folder = "D:/hf-cache"
device = "cuda"
local_files_only = true
```

Indexing targets Qdrant by default. Local keyword-only inspection does not require building the vector index:

```powershell
osu-bot inspect --keyword-only "What is a beatmap?"
```
