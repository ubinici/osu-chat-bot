# osu! Chatbot MVP Roadmap

## Goal

Deploy a complete, understandable osu! question-answering pipeline on a DigitalOcean CPU Droplet before July 31, 2026, evaluate it with real questions, and keep it portable to another host afterward.

## 1. Dense retrieval baseline

- Keep one replaceable dense retrieval backend.
- Split source sections into embedding-sized chunks.
- Store complete chunk payloads in Qdrant so serving does not load source JSONL artifacts.
- Keep intent and osu! alias handling as query hints, not document-routing rules.
- Rebuild into the `osu_wiki_en_dense_v2` collection.
- Run `eval/osu_seed.jsonl` and inspect failures by category.

Release gate:

- Unit suite passes.
- Every retrieved result contains usable text, title, document ID, URL, and score.
- Retrieval improves materially over the saved 27/70 dense baseline.

## 2. Complete response path

- [x] Put generation behind a small provider interface.
- [x] Keep Ollama as the default and support an OpenAI-compatible HTTP provider.
- [x] Require cited, context-grounded answers and a clear insufficient-context response.
- [x] Add liveness, readiness, and chat HTTP endpoints.
- [x] Preserve the CLI for inspection and evaluation.

## 3. DigitalOcean build and deployment

- Use an eligible CPU Droplet; do not rely on GPU or excluded third-party inference credits.
- Run ingestion and indexing on the Droplet with persistent model and Qdrant caches.
- Start with a small quantized instruction model and measure response latency before trying a larger one.
- [x] Package the CPU-only stack with environment variables and persistent volumes.
- Record RAM, latency, retrieval score, and model settings for each serious run.

## 4. Real-world iteration

- Collect anonymized questions, retrieved document IDs, latency, and explicit user feedback.
- Add failed or surprising questions to a versioned evaluation set.
- Fix query aliases, chunking, or prompts only when failures demonstrate the need.
- Add another retrieval backend only if the dense baseline shows a repeatable class of misses.

## 5. Exit before July 31

- Export a Qdrant snapshot and evaluation reports.
- Save deployment configuration without secrets.
- Copy required artifacts off DigitalOcean.
- Destroy all billable Droplets, volumes, and snapshots that should not continue on standard billing.
- Verify the DigitalOcean billing page has no unintended resources.
