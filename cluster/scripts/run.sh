#!/usr/bin/env bash
set -euo pipefail

CLUSTER_USER="${CLUSTER_USER:-${LSV_USER:-}}"
: "${CLUSTER_USER:?Set CLUSTER_USER to your cluster username.}"

TASK="${1:-}"
shift || true

PROJECT_DIR="${PROJECT_DIR:-/nethome/${CLUSTER_USER}/projects/osu-chat-bot}"
CONFIG_PATH="${CONFIG_PATH:-cluster/config.cluster.toml}"
BASE_ARTIFACT_DIR="${BASE_ARTIFACT_DIR:-${PROJECT_DIR}/artifacts/rag}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ARTIFACT_DIR="${RUN_ARTIFACT_DIR:-${PROJECT_DIR}/artifacts/runs/${RUN_ID}/rag}"
EVAL_DATASET="${EVAL_DATASET:-eval/osu_seed.jsonl}"
HF_HOME="${HF_HOME:-/scratch/${CLUSTER_USER}/hf-cache}"
SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-${HF_HOME}/sentence-transformers}"
HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"

export HF_HOME
export SENTENCE_TRANSFORMERS_HOME
export HF_HUB_DISABLE_PROGRESS_BARS
export OSU_BOT_ARTIFACT_SOURCE_PATH="${BASE_ARTIFACT_DIR}"
export OSU_BOT_ARTIFACT_PATH="${RUN_ARTIFACT_DIR}"

mkdir -p "${RUN_ARTIFACT_DIR}"

cd "${PROJECT_DIR}"

echo "Task: ${TASK}"
echo "Cluster user: ${CLUSTER_USER}"
echo "Project: ${PROJECT_DIR}"
echo "Config: ${CONFIG_PATH}"
echo "Base artifacts: ${BASE_ARTIFACT_DIR}"
echo "Run artifacts: ${RUN_ARTIFACT_DIR}"
echo "Run ID: ${RUN_ID}"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none}"
echo "Started: $(date --iso-8601=seconds)"

case "${TASK}" in
  smoke)
    python -m osu_chatbot.app.cli --config "${CONFIG_PATH}" eval "${EVAL_DATASET}" \
      --output "${RUN_ARTIFACT_DIR}/smoke_eval_keyword_report.json"
    ;;

  ner_full)
    python -m osu_chatbot.app.cli --config "${CONFIG_PATH}" entities "$@"
    python -m osu_chatbot.app.cli --config "${CONFIG_PATH}" normalize-entities
    ;;

  entities)
    python -m osu_chatbot.app.cli --config "${CONFIG_PATH}" entities "$@"
    ;;

  normalize_entities)
    python -m osu_chatbot.app.cli --config "${CONFIG_PATH}" normalize-entities
    ;;

  index_interval)
    OFFSET="${1:-0}"
    LIMIT="${2:-2000}"
    BATCH_SIZE="${3:-32}"
    python -m osu_chatbot.app.cli --config "${CONFIG_PATH}" index \
      --offset "${OFFSET}" \
      --limit "${LIMIT}" \
      --batch-size "${BATCH_SIZE}"
    ;;

  eval_keyword)
    python -m osu_chatbot.app.cli --config "${CONFIG_PATH}" eval "${EVAL_DATASET}" \
      --output "${RUN_ARTIFACT_DIR}/eval_seed_keyword_report.json"
    ;;

  eval_dense)
    python -m osu_chatbot.app.cli --config "${CONFIG_PATH}" eval "${EVAL_DATASET}" \
      --dense \
      --output "${RUN_ARTIFACT_DIR}/eval_seed_dense_report.json"
    ;;

  *)
    echo "Unknown task: ${TASK}" >&2
    exit 2
    ;;
esac

echo "Finished: $(date --iso-8601=seconds)"
