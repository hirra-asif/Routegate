#!/usr/bin/env bash
# No API calls are made by this generation wrapper.
# Paths: ROUTEGATE_DATA_DIR (default $HOME/routegate-data) and
# ROUTEGATE_CONTAINER_FILE (default $HOME/containers/vllm-openai-v0.10.1.1.sif).
set -euo pipefail
task_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
# Only $HOME is bound into the container, so use its spelling of the path.
task_home_physical="$(cd "$HOME" && pwd -P)"
case "$task_repo" in "$task_home_physical"/*) task_repo="$HOME/${task_repo#"$task_home_physical"/}" ;; esac
task_container="${ROUTEGATE_CONTAINER_FILE:-$HOME/containers/vllm-openai-v0.10.1.1.sif}"
task_data="${ROUTEGATE_DATA_DIR:-$HOME/routegate-data}"
task_roots=(--data-root "$task_data" --study-root "$task_data/cancellation_controls_v1")
cd "$task_repo"
task_gpu_flags=()
if [[ "${1:-}" == "generate" ]]; then
  task_gpu_flags=(--nv)
  # Required CPU-only hook tests run inside the same container as generation.
  apptainer exec --cleanenv --bind "$HOME:$HOME" "$task_container" \
    "$task_repo/.venv/bin/python" "$task_repo/experiments/cancellation_controls_v1/run_experiment.py" "${task_roots[@]}" selftest
fi
exec apptainer exec ${task_gpu_flags[@]+"${task_gpu_flags[@]}"} --cleanenv --bind "$HOME:$HOME" "$task_container" \
  "$task_repo/.venv/bin/python" "$task_repo/experiments/cancellation_controls_v1/run_experiment.py" "${task_roots[@]}" "$@"
