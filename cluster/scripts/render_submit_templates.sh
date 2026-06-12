#!/usr/bin/env bash
set -euo pipefail

: "${CLUSTER_USER:?Set CLUSTER_USER, for example bualtar.}"
: "${HOME_DIR:?Set HOME_DIR to the absolute home path, usually /nethome/<user>.}"
: "${PROJECT_DIR:?Set PROJECT_DIR to the absolute project path.}"
: "${LOG_DIR:?Set LOG_DIR to the absolute HTCondor log directory.}"
: "${HF_HOME:?Set HF_HOME to the absolute Hugging Face cache directory.}"
: "${DOCKER_IMAGE:?Set DOCKER_IMAGE to the full registry image name.}"

TEMPLATE_DIR="${TEMPLATE_DIR:-cluster/submit}"
OUTPUT_DIR="${OUTPUT_DIR:-cluster/submit/rendered}"

mkdir -p "${OUTPUT_DIR}"

for template in "${TEMPLATE_DIR}"/*.sub; do
  name="$(basename "${template}")"
  sed \
    -e "s|<user>|${CLUSTER_USER}|g" \
    -e "s|<home_dir>|${HOME_DIR}|g" \
    -e "s|<project_dir>|${PROJECT_DIR}|g" \
    -e "s|<log_dir>|${LOG_DIR}|g" \
    -e "s|<hf_home>|${HF_HOME}|g" \
    -e "s|<docker_image>|${DOCKER_IMAGE}|g" \
    "${template}" > "${OUTPUT_DIR}/${name}"
done

echo "Rendered submit files to ${OUTPUT_DIR}"
