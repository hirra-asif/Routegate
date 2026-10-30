#!/usr/bin/env bash
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_repo="$(cd -- "$task_dir/../.." && pwd)"
task_image="$HOME/containers/vllm-openai-v0.10.1.1.sif"
apptainer exec --cleanenv --bind "$HOME:$HOME" "$task_image" \
  "$task_repo/.venv/bin/python" "$task_dir/test_audit.py"
apptainer exec --cleanenv --bind "$HOME:$HOME" "$task_image" \
  "$task_repo/.venv/bin/python" "$task_dir/audit.py" --repo "$task_repo" "$@"
