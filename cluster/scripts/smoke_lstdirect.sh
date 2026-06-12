#!/bin/bash
set -euo pipefail

PROJECT_DIR="/nethome/ubinici/projects/osu-chat-bot"
RUN_ID="${CLUSTER:-manual}.${PROCESS:-0}"
RUN_ARTIFACT_DIR="${PROJECT_DIR}/artifacts/runs/smoke-${RUN_ID}/rag"

export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OSU_BOT_ARTIFACT_SOURCE_PATH="${PROJECT_DIR}/artifacts/rag"
export OSU_BOT_ARTIFACT_PATH="${RUN_ARTIFACT_DIR}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="/nethome/ubinici/scratch/hf-cache"

mkdir -p "${RUN_ARTIFACT_DIR}"

echo "==== smoke preflight ===="
hostname
pwd
python3 --version
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "RUN_ARTIFACT_DIR=${RUN_ARTIFACT_DIR}"
echo "PYTHONPATH=${PYTHONPATH}"

cd "${PROJECT_DIR}"

python3 -m osu_chatbot.app.cli --config cluster/config.cluster.toml eval eval/osu_seed.jsonl \
  --output "${RUN_ARTIFACT_DIR}/smoke_eval_keyword_report.json"
