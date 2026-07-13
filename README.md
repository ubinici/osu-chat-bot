# osu! Chatbot

A small retrieval-augmented chatbot for answering questions about the osu! ecosystem with cited sources.

The current corpus is the English osu! wiki and osu! news archive. Corpus adapters can be added later without changing retrieval or generation.

## Pipeline

```text
offline: osu! sources -> documents -> embedding-sized chunks -> Qdrant
online:  query -> intent and alias hints -> dense retrieval -> grounded prompt -> answer with citations
```

The serving path deliberately uses one retrieval strategy. Qdrant stores the complete chunk payload, so querying does not load the large document and chunk JSONL artifacts into application memory.

## Quick start

Python 3.11 or newer is required.

```powershell
python -m pip install -e ".[dev]"
osu-bot ingest
osu-bot validate
osu-bot index
osu-bot inspect "What does AR change?"
```

Generation currently uses Ollama:

```powershell
ollama pull mistral
ollama serve
osu-bot query "What is osu!direct?"
```

## Commands

- `ingest`: parse configured sources into documents and embedding-sized chunks.
- `validate`: check generated artifacts before indexing.
- `index`: embed chunks and upsert their vectors and complete payloads into Qdrant.
- `inspect`: show query analysis and retrieved chunks without generation.
- `eval`: measure document/chunk retrieval against a JSONL evaluation set.
- `query`: retrieve evidence and ask Ollama for a cited answer.

Additional `terms`, `links`, `entities`, `normalize-entities`, and `stats` commands are offline corpus-analysis utilities. They are not required by the serving path.

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
```

The Qdrant URL can point to embedded storage or a remote service. Environment variables can override deployment-sensitive paths:

- `OSU_BOT_ARTIFACT_PATH`
- `OSU_BOT_ARTIFACT_SOURCE_PATH`
- `OSU_BOT_QDRANT_URL`
- `OSU_BOT_QDRANT_COLLECTION`
- `OSU_BOT_QDRANT_VECTOR_SIZE`

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
```

Expectations use actual corpus document IDs so the metric describes retrieval behavior without hidden topic-routing equivalences.

## Package layout

- `osu_chatbot.corpus`: source parsing and chunk construction.
- `osu_chatbot.indexing`: embeddings and Qdrant indexing.
- `osu_chatbot.retrieval`: intent hints and the replaceable dense backend.
- `osu_chatbot.generation`: grounded prompt and Ollama generation.
- `osu_chatbot.evaluation`: retrieval datasets and metrics.
- `osu_chatbot.quality`: artifact validation and statistics.
- `osu_chatbot.app`: CLI entry point.

The guided Colab workflow is in `notebooks/osu_chatbot_colab_walkthrough.ipynb`.
