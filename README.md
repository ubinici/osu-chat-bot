# osu! Chatbot

A small retrieval-augmented chatbot for answering questions about the osu! ecosystem with cited sources.

The current corpus is the English osu! wiki and osu! news archive. Corpus adapters can be added later without changing retrieval or generation.

## Pipeline

```text
offline: source adapters -> normalized documents + aliases -> embedding-sized chunks -> Qdrant
online:  Discord/API -> FastAPI -> query analysis -> retrieval -> GPT-OSS -> cited answer
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

Generation uses the Ollama API. The deployed stack is configured for GPT-OSS through Ollama Cloud:

```powershell
set OSU_BOT_GENERATION_URL=https://ollama.com
set OSU_BOT_GENERATION_MODEL=gpt-oss:20b
set OSU_BOT_GENERATION_API_KEY=your-secret
set OSU_BOT_GENERATION_THINK=low
osu-bot query "What is osu!direct?"
```

Run the HTTP API:

```powershell
osu-bot serve
```

It exposes `GET /healthz`, `GET /readyz`, and `POST /v1/chat`.
The chat route is async and offloads the blocking RAG call from the event loop. Set
`OSU_BOT_MAX_CONCURRENT_REQUESTS` to bound concurrent model calls; the cloud deployment
starts at `4`, while a local CPU model should usually start at `1`.

Run the Discord slash-command client against that API:

```powershell
python -m pip install -e ".[discord]"
set OSU_BOT_DISCORD_TOKEN=your-bot-token
set OSU_BOT_DISCORD_GUILD_ID=your-test-server-id
osu-bot discord
```

Use `/initiate` in a server to create a private room, `/ask` inside that room, and `/close`
to finish early. The bot accepts questions only from the room owner, caps the number of
active rooms, and removes idle rooms after five minutes by default. Each room gets isolated,
bounded conversation context. Finalized transcripts are stored in
`artifacts/feedback/discord_sessions.sqlite3`; feedback is appended to
`artifacts/feedback/events.jsonl`. Neither store contains Discord user IDs.

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
- `discord`: run the Discord private-room client against the HTTP API.
- `build-style-profile`: reduce a JSONL chat export to aggregate style statistics.

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
url = "https://ollama.com"
model = "gpt-oss:20b"
think = "low"
max_concurrent_requests = 4
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
- `OSU_BOT_MAX_CONCURRENT_REQUESTS`
- `OSU_BOT_STYLE_PROFILE_PATH`
- `OSU_BOT_DISCORD_TOKEN`
- `OSU_BOT_DISCORD_API_URL`
- `OSU_BOT_DISCORD_GUILD_ID`
- `OSU_BOT_DISCORD_FEEDBACK_PATH`
- `OSU_BOT_DISCORD_EPHEMERAL`
- `OSU_BOT_ANSWER_VERSION`
- `OSU_BOT_DISCORD_MAX_ACTIVE_ROOMS`
- `OSU_BOT_DISCORD_INACTIVITY_SECONDS`
- `OSU_BOT_DISCORD_CONTEXT_TURNS`
- `OSU_BOT_DISCORD_SESSION_DB_PATH`
- `OSU_BOT_DISCORD_CATEGORY_ID`

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

## Aggregate response style

The default prompt now aims for a casual, concise, friendly osu! voice without forcing slang.
You can derive soft style tendencies from a JSONL chat export whose message field is `content`:

```powershell
osu-bot build-style-profile private/chat.jsonl `
  --output artifacts/style/osu_chat_style.json `
  --minimum-messages 100
set OSU_BOT_STYLE_PROFILE_PATH=artifacts/style/osu_chat_style.json
```

The generated artifact contains only counts and ratios: message length, punctuation,
lowercase usage, emoticon usage, and frequencies for a small predefined marker list. It
does not retain quotes, n-grams, usernames, or user IDs, and the prompt treats the result
as a soft tendency rather than an instruction to imitate an individual. Only process chat
data you are permitted to use; keep raw exports out of the repository.

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
