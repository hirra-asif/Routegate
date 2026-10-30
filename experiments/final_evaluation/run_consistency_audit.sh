#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="${ROUTEGATE_DATA_DIR:-$HOME/routegate-data}"
CONTAINER_FILE="${ROUTEGATE_CONTAINER_FILE:-$HOME/containers/vllm-openai-v0.10.1.1.sif}"

test -f "$CONTAINER_FILE"
test -f "$REPO_DIR/.venv/bin/python"
apptainer exec --cleanenv --bind "$HOME:$HOME" \
  "$CONTAINER_FILE" "$REPO_DIR/.venv/bin/python" \
  "$SCRIPT_DIR/audit_final_consistency.py" --repo "$REPO_DIR" --data-root "$DATA_DIR"

apptainer exec --cleanenv --bind "$HOME:$HOME" \
  "$CONTAINER_FILE" "$REPO_DIR/.venv/bin/python" \
  "$SCRIPT_DIR/compare_steering_arithmetic.py" --repo "$REPO_DIR" --data-root "$DATA_DIR" "$@"

ARCHIVE_PATH="$(mktemp "$HOME/routegate_consistency_audit_XXXXXX.tar.gz")"
tar -czf "$ARCHIVE_PATH" -C "$REPO_DIR/outputs/thesis_results" consistency_review_v1
printf 'Audit archive: %s\n' "$ARCHIVE_PATH"
