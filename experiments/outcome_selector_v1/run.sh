#!/usr/bin/env bash
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_repo="$(cd -- "$task_dir/../.." && pwd)"
task_stage="${1:-run}"
if [ "$#" -gt 0 ]; then shift; fi
case "$task_stage" in
  preflight|run|status|selftest) ;;
  *) echo "Usage: bash experiments/outcome_selector_v1/run.sh {preflight|run|status|selftest} [options]" >&2; exit 2 ;;
esac
task_image="$HOME/containers/vllm-openai-v0.10.1.1.sif"
test -f "$task_image" || { echo "Missing container: $task_image" >&2; exit 1; }
task_python="$task_repo/.venv/bin/python"
if [ "$task_stage" = selftest ]; then
  apptainer exec --cleanenv --bind "$HOME:$HOME" "$task_image" \
    /usr/bin/env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
    "$task_python" "$task_dir/test_selector.py"
else
  apptainer exec --cleanenv --bind "$HOME:$HOME" "$task_image" \
    /usr/bin/env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
    "$task_python" "$task_dir/run_comparison.py" "$task_stage" --repo "$task_repo" "$@"
fi
