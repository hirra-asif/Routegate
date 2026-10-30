#!/usr/bin/env bash
# Rebuild the final-evaluation report from completed records inside the project
# container. Override the data root or image with ROUTEGATE_DATA_DIR and
# ROUTEGATE_CONTAINER_FILE.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/../.." && pwd)
# Only $HOME is bound into the container, so use its spelling of the path.
HOME_PHYSICAL="$(cd "$HOME" && pwd -P)"
case "$REPO_DIR" in "$HOME_PHYSICAL"/*) REPO_DIR="$HOME/${REPO_DIR#"$HOME_PHYSICAL"/}" ;; esac
SCRIPT_DIR="$REPO_DIR/experiments/final_evaluation"
DATA_DIR="${ROUTEGATE_DATA_DIR:-$HOME/routegate-data}"
CONTAINER_FILE="${ROUTEGATE_CONTAINER_FILE:-$HOME/containers/vllm-openai-v0.10.1.1.sif}"
test -f "$CONTAINER_FILE"
cd "$REPO_DIR"
apptainer exec --cleanenv --bind "$HOME:$HOME" \
  "$CONTAINER_FILE" \
  "$REPO_DIR/.venv/bin/python" \
  "$SCRIPT_DIR/build_final_thesis_results.py" \
  --repo "$REPO_DIR" --data-root "$DATA_DIR" "$@"
