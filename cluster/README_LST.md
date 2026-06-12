# LST/coli HTCondor Workflow

Use this when your account belongs to the LST/coli cluster convention instead of the LSV one.

Do not mix hostnames and registries between conventions:

- LST/coli login: `login.lst.uni-saarland.de`
- LST/coli submit: `ssh submit` from the login host, or `submit.coli.uni-saarland.de` when reachable directly
- LST/coli Docker registry: `docker.coli.uni-saarland.de`
- project/home storage: `/nethome/$USER`, backed up, quota-limited
- logs and larger temporary/cache data: `/scratch/$USER`, not backed up

The project still uses the same Python package and cluster scripts as the LSV workflow.

Important Docker rule from the LST wiki: for Docker jobs, `initialdir` must be your home directory, not a project subdirectory. The submit templates follow this by using `<home_dir>` as `initialdir`, while the wrapper script `cd`s into `<project_dir>`.

## 1. Copy Project

From your local machine:

```powershell
ssh bualtar@login.lst.uni-saarland.de "mkdir -p ~/projects"
scp .\osu-chat-bot-cluster.tgz bualtar@login.lst.uni-saarland.de:~/projects/
```

On LST:

```bash
ssh bualtar@login.lst.uni-saarland.de
mkdir -p ~/projects
cd ~/projects
tar -xzf osu-chat-bot-cluster.tgz
cd osu-chat-bot
chmod +x cluster/scripts/run.sh cluster/scripts/index_interval.sh cluster/scripts/render_submit_templates.sh
```

## 2. Choose Paths

Use the actual paths reported by your shell:

```bash
export CLUSTER_USER=bualtar
export HOME_DIR="$HOME"
export PROJECT_DIR="$(pwd)"
export LOG_DIR="/scratch/$USER/logs/osu-chat-bot/logfiles"
export HF_HOME="/scratch/$USER/hf-cache"
export DOCKER_IMAGE="docker.coli.uni-saarland.de/bualtar/osu-chat-bot-rag:v1"

mkdir -p "$LOG_DIR" "$HF_HOME"
```

Check storage before large jobs:

```bash
quota -s
condor_nodestate
```

## 3. Create Config

```bash
cp cluster/config.cluster.toml.example cluster/config.cluster.toml
nano cluster/config.cluster.toml
```

Use `PROJECT_DIR` and `HF_HOME` values from above:

```toml
[corpus]
osu_wiki_path = "/absolute/path/to/osu-chat-bot/database/osu-wiki"
language = "en"
include_news = true

[artifacts]
source_path = "/absolute/path/to/osu-chat-bot/artifacts/rag"
path = "/absolute/path/to/osu-chat-bot/artifacts/runs/manual/rag"

[embedding]
model = "sentence-transformers/all-MiniLM-L6-v2"
cache_folder = "/absolute/path/to/hf-cache"
device = "cpu"
local_files_only = false

[qdrant]
url = "http://<qdrant-host>:6333"
collection = "osu_wiki_en_metadata_text_v1_minilm"
vector_size = 384
```

## 4. Build And Push Docker Image

Before pushing for the first time, open `https://docker.coli.uni-saarland.de`, log in with your LST credentials, create a project, and make it public if you want Condor to pull without authentication. The image name must use that project name:

```text
docker.coli.uni-saarland.de/<project_name>/<image_name>:<tag>
```

Run this where Docker build/push is allowed. If login/submit nodes disallow Docker builds, build locally or on an approved workstation, then push to the registry:

```bash
cd "$PROJECT_DIR"
docker login docker.coli.uni-saarland.de
docker build -f cluster/Dockerfile -t "$DOCKER_IMAGE" .
docker push "$DOCKER_IMAGE"
```

## 5. Render Submit Files

```bash
cd "$PROJECT_DIR"
cluster/scripts/render_submit_templates.sh
grep -R "<user>\|<home_dir>\|<project_dir>\|<log_dir>\|<hf_home>\|<docker_image>" cluster/submit/rendered || true
```

The rendered files live in:

```text
cluster/submit/rendered/
```

## 6. Submit Jobs

Submit from the LST submit host:

```bash
ssh bualtar@submit.coli.uni-saarland.de
cd /absolute/path/to/osu-chat-bot
condor_submit cluster/submit/rendered/smoke.sub
condor_q bualtar
```

If direct SSH to the submit host is not available:

```bash
ssh bualtar@login.lst.uni-saarland.de
ssh submit
cd /absolute/path/to/osu-chat-bot
condor_submit cluster/submit/rendered/smoke.sub
```

Monitor and debug:

```bash
condor_q
condor_q -allusers
condor_q -better-analyze <JOBID>
condor_q -hold <JOBID>
condor_ssh_to_job <JOBID>
```

Use `condor_ssh_to_job` only to inspect a running job; do not start extra workloads inside it.

After smoke succeeds:

```bash
condor_submit cluster/submit/rendered/ner_full.sub
python cluster/scripts/make_offsets.py --limit 2000 --batch-size 32 > cluster/submit/index_offsets.txt
condor_submit cluster/submit/rendered/index_many.sub
condor_submit cluster/submit/rendered/eval_keyword.sub
condor_submit cluster/submit/rendered/eval_dense.sub
```

Dense indexing/eval requires a real Qdrant server URL in `cluster/config.cluster.toml`.

## Flocking Note

The LST and LSV pools can flock jobs between clusters. Our rendered submit templates restrict jobs to the LST/coli filesystem domain:

```condor
requirements = (TARGET.UidDomain == "coli.uni-saarland.de")
```

Keep this while the job depends on `/nethome` and `/scratch` paths. If you intentionally allow flocking, you must add proper `should_transfer_files`, `transfer_input_files`, and `transfer_output_files` rules.

## Webservice/Qdrant Note

The LST wiki supports services inside Docker jobs with:

```condor
docker_network_type = host
container_service_names = SERVICE_NAME
SERVICE_NAME_container_port = SERVICE_PORT
```

This is relevant if you run Qdrant itself as a Condor webservice. For our first cluster attempt, prefer an already reachable Qdrant server URL. If you launch Qdrant through Condor, use `condor_q -run <JOBID>` to find the execute machine, then set:

```toml
[qdrant]
url = "http://<execute-machine>:6333"
```
