# osu! Chatbot Cluster Readiness Roadmap

## Goal

Make the RAG training, indexing, and evaluation pipeline reproducible enough to run on cluster, compare changes, and decide whether a model/retrieval update is actually better.

## Phase 1: Reproducible Local Baseline

- Install from the project package in the active environment: `python -m pip install -e ".[dev,entities]"`.
- Verify local health:
  - `python -m pytest --basetemp=.pytest_tmp`
  - `python -m osu_chatbot.app.cli stats`
  - `python -m osu_chatbot.app.cli validate`
  - `python -m osu_chatbot.app.cli eval eval/osu_seed.jsonl --output artifacts/rag/eval_seed_keyword_report.json`
- Fix environment drift before cluster work. The package declares `qdrant-client`, but dense eval currently fails when the active environment cannot import it.

## Phase 2: Full Knowledge Build

- Run the deterministic corpus pipeline: `ingest`, `terms`, `links`, `stats`, `validate`.
- Run the generative NER pipeline over the full chunk artifact, not just the current 100-chunk pilot:
  - `python -m osu_chatbot.app.cli entities --backend gliner --label-profile main-page --threshold 0.5`
  - `python -m osu_chatbot.app.cli normalize-entities`
- Review `entity_normalization_review.csv`, then decide which accepted/reviewed normalized aliases should be promoted into retrieval.
- Full cluster NER should use run-isolated artifact paths. The cluster wrapper now writes NER outputs under `artifacts/runs/<run_id>/rag` while reading stable prepared inputs from `artifacts/rag`.

## Phase 3: Retrieval Improvements

- Improve embedding inputs before reindexing: include title, heading path, tags, domain, subculture, and chunk text in the embedded text instead of raw chunk text only. Current implementation version: `metadata_text_v1`.
- Add intent coverage for beginner, rules/moderation, mapping, ranking/pp, client settings, account/access, and performance troubleshooting.
- Calibrate retrieval scoring by query type; avoid relying on a simple sum of dense, keyword, entity, and document scores for every query.

## Phase 4: Dense Indexing On Cluster

- Use `cluster/config.cluster.toml` with a server Qdrant URL for parallel jobs. Do not use file-based Qdrant for parallel HTCondor indexing.
- Use `cluster/scripts/run.sh` for cluster tasks. It sets `OSU_BOT_ARTIFACT_SOURCE_PATH`, `OSU_BOT_ARTIFACT_PATH`, and `RUN_ID` so runs do not overwrite each other.
- Prime the Hugging Face cache once with `local_files_only = false`, then switch production jobs to `local_files_only = true`.
- Generate index intervals with `cluster/scripts/make_offsets.py`, submit with the HTCondor files, and save `index_report.json` plus logs for every run.
- Rebuild the dense index after embedding input changes; existing local Qdrant data was built with the previous raw-text-only embedding input.

## Phase 5: Evaluation Gate

- Run keyword and dense retrieval evals on the same dataset and save JSON reports.
- Compare global accuracy, category accuracy, and failed examples from `top_sources`.
- Only then compare generator/model changes, because poor retrieval will mask whether the chat model improved.

## Missing Or Easy To Forget

- A versioned eval dataset and report artifacts for every serious run.
- A clear artifact naming convention for NER, normalized entities, indexes, and eval reports.
- Dependency parity between local and cluster environments.
- Qdrant server availability and collection reset policy.
- A promotion path from normalized generative entities into the runtime term/entity layer.
- A final answer-quality eval after retrieval eval, ideally checking citation support and refusal behavior.
