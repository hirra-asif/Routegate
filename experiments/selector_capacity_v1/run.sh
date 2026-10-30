#!/usr/bin/env bash
# Linear versus one-hidden-layer selection of the same fixed repair edit.
# CPU only, one thread, no GPU, no network, no new generations.
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_repo="$(cd -- "$task_dir/../.." && pwd)"
task_base="$task_repo/experiments/outcome_selector_v1"
task_stage="${1:-run}"
if [ "$#" -gt 0 ]; then shift; fi
case "$task_stage" in
  preflight|run|status|selftest) ;;
  *) echo "Usage: bash experiments/selector_capacity_v1/run.sh {preflight|run|status|selftest} [options]" >&2; exit 2 ;;
esac
test -d "$task_base" || { echo "Missing the reused experiment: $task_base" >&2; exit 1; }
task_image="$HOME/containers/vllm-openai-v0.10.1.1.sif"
test -f "$task_image" || { echo "Missing container: $task_image" >&2; exit 1; }
task_python="$task_repo/.venv/bin/python"
task_env=(OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
          VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
          "PYTHONPATH=$task_dir:$task_base")
if [ "$task_stage" = selftest ]; then
  apptainer exec --cleanenv --bind "$HOME:$HOME" "$task_image" \
    /usr/bin/env "${task_env[@]}" "$task_python" "$task_dir/test_capacity.py"
else
  apptainer exec --cleanenv --bind "$HOME:$HOME" "$task_image" \
    /usr/bin/env "${task_env[@]}" "$task_python" "$task_dir/run_capacity.py" "$task_stage" --repo "$task_repo" "$@"
fi
