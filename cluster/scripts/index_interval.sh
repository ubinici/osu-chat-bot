#!/usr/bin/env bash
set -euo pipefail

: "${LSV_USER:?Set LSV_USER to your LSV username.}"

PROJECT_DIR="${PROJECT_DIR:-/nethome/${LSV_USER}/projects/osu-chat-bot}"
CONFIG_PATH="${CONFIG_PATH:-cluster/config.cluster.toml}"
BASE_ARTIFACT_DIR="${BASE_ARTIFACT_DIR:-${PROJECT_DIR}/artifacts/rag}"
RUN_ID="${RUN_ID:-index-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ARTIFACT_DIR="${RUN_ARTIFACT_DIR:-${PROJECT_DIR}/artifacts/runs/${RUN_ID}/rag}"
HF_HOME="${HF_HOME:-/data/users/${LSV_USER}/hf-cache}"
SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-${HF_HOME}/sentence-transformers}"
HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"

export HF_HOME
export SENTENCE_TRANSFORMERS_HOME
export HF_HUB_DISABLE_PROGRESS_BARS
export OSU_BOT_ARTIFACT_SOURCE_PATH="${BASE_ARTIFACT_DIR}"
export OSU_BOT_ARTIFACT_PATH="${RUN_ARTIFACT_DIR}"

mkdir -p "${RUN_ARTIFACT_DIR}"

OFFSET="${1:-${INDEX_OFFSET:-0}}"
LIMIT="${2:-${INDEX_LIMIT:-2000}}"
BATCH_SIZE="${3:-${INDEX_BATCH_SIZE:-32}}"

cd "${PROJECT_DIR}"

echo "Project: ${PROJECT_DIR}"
echo "Config: ${CONFIG_PATH}"
echo "HF_HOME: ${HF_HOME}"
echo "Base artifacts: ${BASE_ARTIFACT_DIR}"
echo "Run artifacts: ${RUN_ARTIFACT_DIR}"
echo "Offset: ${OFFSET}"
echo "Limit: ${LIMIT}"
echo "Batch size: ${BATCH_SIZE}"
echo "Started: $(date --iso-8601=seconds)"

python -m osu_chatbot.app.cli --config "${CONFIG_PATH}" index \
  --offset "${OFFSET}" \
  --limit "${LIMIT}" \
  --batch-size "${BATCH_SIZE}"

echo "Finished: $(date --iso-8601=seconds)"
