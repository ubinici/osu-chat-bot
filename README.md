# osu! Chatbot

A small retrieval-augmented chatbot for answering questions about the osu! ecosystem with cited sources.

The current corpus is the English osu! wiki and osu! news archive. Corpus adapters can be added later without changing retrieval or generation.

## Pipeline

```text
offline: source adapters -> normalized documents + aliases -> embedding-sized chunks -> Qdrant
online:  query -> query analysis -> retrieval policy -> dense retrieval -> grounded answer with citations
```

The serving path deliberately uses one retrieval strategy. Qdrant stores the complete chunk payload, so querying does not load the large document and chunk JSONL artifacts into application memory.

## Quick start

Python 3.11 or newer is required.

```powershell
python -m pip install -e ".[dev]"
osu-bot ingest
osu-bot links
osu-bot aliases
osu-bot validate
osu-bot index
osu-bot inspect "What does AR change?"
```

Generation defaults to Ollama:

```powershell
ollama pull mistral
ollama serve
osu-bot query "What is osu!direct?"
```

Run the HTTP API:

```powershell
osu-bot serve
```

It exposes `GET /healthz`, `GET /readyz`, and `POST /v1/chat`.

## Commands

- `ingest`: parse configured sources into documents and embedding-sized chunks.
- `links`: derive reviewable alias evidence from document links.
- `aliases`: build the compact runtime topic-alias artifact.
- `validate`: check generated artifacts before indexing.
- `index`: embed chunks and upsert their vectors and complete payloads into Qdrant.
- `inspect`: show query analysis and retrieved chunks without generation.
- `eval`: measure document/chunk retrieval against a JSONL evaluation set.
- `query`: retrieve evidence and ask the configured generator for a cited answer.
- `serve`: run the minimal HTTP chat API with one inference worker.

Additional `terms`, `entities`, `normalize-entities`, and `stats` commands are offline corpus-analysis utilities. They are not required by the serving path.

## Configuration

The default configuration lives in `config.toml`.

```toml
[embedding]
model = "sentence-transformers/all-MiniLM-L6-v2"

[qdrant]
url = "file://artifacts/rag/qdrant"
collection = "osu_wiki_en_dense_v2"
vector_size = 384

[retrieval]
top_k = 6
alias_artifact = "document_aliases.jsonl"
alias_minimum_confidence = 0.85
alias_minimum_tokens = 2
preferred_document_limit = 4
soft_preferred_document_limit = 1
canonical_source_types = ["wiki"]
temporal_source_types = ["wiki", "news"]
excluded_chunk_types = ["citation", "formula"]

[generation]
provider = "ollama"
url = "http://127.0.0.1:11434"
model = "mistral"
```

The Qdrant URL can point to embedded storage or a remote service. Environment variables can override deployment-sensitive paths:

- `OSU_BOT_ARTIFACT_PATH`
- `OSU_BOT_ARTIFACT_SOURCE_PATH`
- `OSU_BOT_QDRANT_URL`
- `OSU_BOT_QDRANT_COLLECTION`
- `OSU_BOT_QDRANT_VECTOR_SIZE`
- `OSU_BOT_GENERATION_PROVIDER`
- `OSU_BOT_GENERATION_URL`
- `OSU_BOT_GENERATION_MODEL`
- `OSU_BOT_GENERATION_API_KEY`
- `OSU_BOT_GENERATION_TEMPERATURE`
- `OSU_BOT_GENERATION_TIMEOUT_SECONDS`

Supported generation providers are `ollama` and `openai-compatible`. For the latter, configure the base URL through `/v1`; the adapter calls its `/chat/completions` endpoint.

## Chunking and indexing

Section content is split with a conservative 180-token estimate and 24-token overlap before embedding. Titles, heading paths, tags, source information, and document IDs are added to the embedding input.

Indexing is resumable:

```powershell
osu-bot index --limit 2000 --batch-size 32
osu-bot index --resume --limit 2000 --batch-size 32
```

Re-run `ingest` and rebuild the collection whenever chunking or the embedding-input version changes. Point IDs are deterministic, but using a fresh collection name makes evaluation comparisons easier.

## Evaluation

The seed set contains direct terminology, colloquial questions, and support symptoms:

```powershell
osu-bot eval eval/osu_seed.jsonl
osu-bot eval eval/osu_seed.jsonl --output artifacts/rag/eval_dense_report.json
osu-bot eval eval/osu_seed.jsonl --top-k 10
```

Build a reviewed query-to-topic dataset from future serving feedback while excluding exact benchmark-query overlap:

```powershell
osu-bot build-topic-dataset artifacts/feedback/events.jsonl artifacts/feedback/reviews.jsonl `
  --output artifacts/learning/query_topics_feedback.jsonl `
  --held-out eval/osu_seed.jsonl
```

The initial reviewed paraphrase set and feedback schemas live in [training/README.md](training/README.md).

Compare a semantic query-topic baseline with the deterministic alias resolver:

```powershell
osu-bot eval-topic-model training/query_topics_seed.jsonl `
  --alias-artifact artifacts/rag/alias_build_check/document_aliases.jsonl
```

The chat response also has an explicit `response_type`. Very small, high-confidence cases of missing context return `clarification` with a structured prompt and options, without calling retrieval or generation.

Reports include Hit@1, Hit@3, Hit@6, mean reciprocal rank, resolved topics, and retrieval lanes. Expectations use actual corpus document IDs so the metric describes retrieval behavior without hidden topic-routing equivalences.

The source-neutral record, alias, retrieval, future tool-routing, and feedback contracts are documented in [docs/architecture.md](docs/architecture.md).

## Package layout

- `osu_chatbot.corpus`: source parsing and chunk construction.
- `osu_chatbot.indexing`: embeddings and Qdrant indexing.
- `osu_chatbot.retrieval`: replaceable query analysis, retrieval policy, and dense backend.
- `osu_chatbot.generation`: grounded prompt and replaceable generation providers.
- `osu_chatbot.evaluation`: retrieval datasets and metrics.
- `osu_chatbot.learning`: reviewed feedback and query-to-topic dataset contracts.
- `osu_chatbot.quality`: artifact validation and statistics.
- `osu_chatbot.app`: CLI entry point.

The guided Colab workflow is in `notebooks/osu_chatbot_colab_walkthrough.ipynb`.

The CPU-only DigitalOcean deployment workflow is in `deploy/README.md`.
