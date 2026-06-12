# HTCondor Cluster Workflow

This folder contains a cluster-oriented scaffold for full osu! RAG runs: smoke tests, full generative NER, dense Qdrant indexing, and retrieval evaluation.

The templates currently follow the LSV guide shape:

- project files live under `/nethome/<user>/projects/osu-chat-bot`
- logs live under `/data/users/<user>/logs/osu-chat-bot/logfiles`
- Docker images are built/pushed from `ws71lx.lsv.uni-saarland.de`
- jobs are submitted from `submit.lsv.uni-saarland.de`

If your account uses the LST/coli convention instead, change all matching hostnames, registry names, and paths together. Do not mix LSV and LST/coli conventions in one setup:

- LSV-style examples use `contact.lsv.uni-saarland.de`, `submit.lsv.uni-saarland.de`, and `docker.lsv.uni-saarland.de`.
- LST/coli-style examples may use `login.lst.uni-saarland.de`, `submit.coli.uni-saarland.de`, and `docker.coli.uni-saarland.de`.

For LST/coli-specific commands, see `cluster/README_LST.md`.

Do not commit tokens, passwords, or private hostnames.

## Workflow Shape

Every cluster experiment should have:

- a package command, routed through `cluster/scripts/run.sh`
- a config file, usually `cluster/config.cluster.toml`
- a submit file under `cluster/submit/`
- a unique `RUN_ID`
- logs in the cluster log directory
- JSON reports or artifacts in `artifacts/runs/<run_id>/rag`

We keep the existing Python package layout (`src/osu_chatbot`) instead of moving logic into loose one-off scripts. The cluster wrapper simply dispatches to `python -m osu_chatbot.app.cli`.

Submit files are templates. Render them for your cluster with `cluster/scripts/render_submit_templates.sh` after setting `CLUSTER_USER`, `PROJECT_DIR`, `LOG_DIR`, `HF_HOME`, and `DOCKER_IMAGE`.

## 1. Prepare Local Artifacts

Run this locally before copying to the cluster:

```powershell
python -m osu_chatbot.app.cli ingest
python -m osu_chatbot.app.cli terms
python -m osu_chatbot.app.cli stats
python -m osu_chatbot.app.cli validate
```

Dense indexing only needs the code plus `artifacts/rag/chunks_hierarchical.jsonl`. Keeping `terms.json` and reports nearby is useful for inspection.

## 2. Copy Project To LSV

On your own machine:

```bash
scp -r /path/to/osu-chat-bot <user>@contact.lsv.uni-saarland.de:/nethome/<user>/projects
```

Avoid copying local vector stores, caches, virtualenvs, and `__pycache__` directories if possible.

On `contact.lsv.uni-saarland.de`, create log/cache directories:

```bash
mkdir -p /nethome/<user>/projects
mkdir -p /data/users/<user>/logs/osu-chat-bot/logfiles
mkdir -p /data/users/<user>/hf-cache
```

## 3. Configure Cluster Paths

Copy the example config and edit placeholders:

```bash
cd /nethome/<user>/projects/osu-chat-bot
cp cluster/config.cluster.toml.example cluster/config.cluster.toml
nano cluster/config.cluster.toml
```

Important fields:

- `artifacts.source_path`: should point to the stable prepared artifact directory on `/nethome`
- `artifacts.path`: is the per-run output location; cluster submit files usually override it with `OSU_BOT_ARTIFACT_PATH`
- `embedding.cache_folder`: should point to a persistent shared cache, usually under `/data/users/<user>/hf-cache`
- `qdrant.url`: should point at a Qdrant server, not `file://`, when running multiple HTCondor jobs

Cluster jobs use `cluster/scripts/run.sh`, which sets:

- `OSU_BOT_ARTIFACT_SOURCE_PATH`: stable input artifacts, usually `artifacts/rag`
- `OSU_BOT_ARTIFACT_PATH`: run output artifacts, usually `artifacts/runs/<run_id>/rag`
- `RUN_ID`: a unique run name used to avoid overwriting NER, index, and eval outputs

## 4. Build And Push Docker Image

From `contact`, SSH to the Docker build workstation:

```bash
ssh <user>@ws71lx.lsv.uni-saarland.de
cd /nethome/<user>/projects/osu-chat-bot
docker build -f cluster/Dockerfile -t docker.lsv.uni-saarland.de/<user>/osu-chat-bot-rag:v1 .
docker push docker.lsv.uni-saarland.de/<user>/osu-chat-bot-rag:v1
logout
```

If you need CUDA/GPU embeddings, adjust `cluster/Dockerfile` to use a CUDA/PyTorch base image that matches the cluster driver/runtime.
After dependency changes, rebuild and push the Docker image again, then make sure all submit files reference the new image tag.

## 5. Prime Hugging Face Cache

Use a read-only HF token. Export it in the shell before submitting; do not write it into files:

```bash
export HF_TOKEN=hf_...
```

For the first cache-priming run, set this in `cluster/config.cluster.toml`:

```toml
[embedding]
local_files_only = false
```

Then submit a tiny interval:

```bash
cd /nethome/<user>/projects/osu-chat-bot
condor_submit cluster/submit/index_interval.sub \
  -append "arguments = 0 10 2"
```

After the model is cached, switch to:

```toml
[embedding]
local_files_only = true
```

This makes production jobs fail fast if the cache is missing instead of repeatedly trying to download from Hugging Face.

## 6. Generate Index Intervals

Create an offset file for interval jobs:

```bash
cd /nethome/<user>/projects/osu-chat-bot
python cluster/scripts/make_offsets.py --limit 2000 --batch-size 32 > cluster/submit/index_offsets.txt
```

For about 42k chunks, `--limit 2000` creates roughly 22 jobs.

## 7. Submit Indexing Jobs

Edit `cluster/submit/index_many.sub` and replace `<user>` plus the Docker image name.

Submit from `submit.lsv.uni-saarland.de`:

```bash
ssh <user>@submit.lsv.uni-saarland.de
cd /nethome/<user>/projects/osu-chat-bot
export HF_TOKEN=hf_...
condor_submit cluster/submit/index_many.sub
watch condor_q <user>
```

Useful monitoring commands:

```bash
condor_q <user>
condor_q -better-analyze <job-id>
condor_q -hold <job-id>
condor_rm <job-id>
```

Logs go to:

```bash
/data/users/<user>/logs/osu-chat-bot/logfiles
```

Read `.out`, `.err`, and `.log` files for each job. The `.out` file will show per-batch progress, embedding time, Qdrant upsert time, ETA, and `next_offset`.

The error file being empty is a good sign, but always inspect all three files after a new workflow or Docker image change.

## 8. Submit Full NER

Run full generative NER into an isolated run directory:

```bash
condor_submit cluster/submit/ner_full.sub
```

The output candidate and normalization artifacts will be written under:

```bash
artifacts/runs/ner-full-<cluster>-<process>/rag
```

After the job completes, review `entity_normalization_review.csv` from that run before promoting aliases into runtime retrieval.

## 9. Run Evaluation

Run retrieval evaluation after indexing:

```bash
condor_submit cluster/submit/eval_keyword.sub
condor_submit cluster/submit/eval_dense.sub
```

Each job writes a category-aware JSON report to its run artifact directory.

## 10. Recommended Execution Order

Use separate jobs and inspect logs after each one:

```bash
# 1. Confirm Docker image, paths, config, and artifact reads.
condor_submit cluster/submit/smoke.sub

# 2. Run full generative NER and normalization into an isolated run.
condor_submit cluster/submit/ner_full.sub

# 3. Rebuild dense index after embedding input/model changes.
python cluster/scripts/make_offsets.py --limit 2000 --batch-size 32 > cluster/submit/index_offsets.txt
condor_submit cluster/submit/index_many.sub

# 4. Evaluate retrieval.
condor_submit cluster/submit/eval_keyword.sub
condor_submit cluster/submit/eval_dense.sub
```

Intent-classifier training should become a separate workflow later, once there are enough labeled intent examples. For now, retrieval, NER coverage, and dense indexing are the higher-leverage cluster jobs.

## 11. Qdrant Note

Parallel HTCondor indexing should write to a Qdrant server:

```toml
[qdrant]
url = "http://<qdrant-host>:6333"
```

Do not run multiple jobs against:

```toml
url = "file://artifacts/rag/qdrant"
```

File-based Qdrant is only suitable for local development or a single sequential job.

## Evaluation Targets

Current retrieval eval reports:

- global retrieval accuracy
- category-level accuracy
- top retrieved documents/chunks and component scores per example

Future evaluation work should add Recall@1/3/5, MRR@10, and nDCG@10. Once generation is enabled in the cluster flow, add groundedness, citation correctness, and unsupported-query/fallback accuracy.

## Details Still Needed

To make this exact for your environment, collect these from your cluster/admin docs:

- the correct Docker/HTCondor submit syntax if it differs from `universe = docker`
- whether workers can reach the Qdrant host
- whether GPU jobs are allowed and the exact `request_gpus`/constraint syntax
- memory/runtime limits for jobs
- whether outbound internet is allowed from workers
- preferred shared cache path
