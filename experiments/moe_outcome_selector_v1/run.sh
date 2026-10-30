#!/usr/bin/env bash
# Residual versus native MoE routing features for selecting the frozen THREE-LAYER static repair.
# CPU only, one numerical thread, no GPU, no model load, no network, no generation, no API call.
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_repo="$(cd -- "$task_dir/../.." && pwd)"
task_base="$task_repo/experiments/outcome_selector_v1"
task_stage="${1:-run}"
if [ "$#" -gt 0 ]; then shift; fi
case "$task_stage" in
  preflight|run|status|selftest) ;;
  *) echo "Usage: bash experiments/moe_outcome_selector_v1/run.sh {preflight|run|status|selftest} [--new-version]" >&2; exit 2 ;;
esac
test -d "$task_base" || { echo "Missing the reused experiment: $task_base" >&2; exit 1; }
task_image="${ROUTEGATE_CONTAINER_FILE:-$HOME/containers/vllm-openai-v0.10.1.1.sif}"
test -f "$task_image" || { echo "Missing container: $task_image" >&2; exit 1; }
task_python="$task_repo/.venv/bin/python"
test -x "$task_python" || { echo "Missing interpreter: $task_python (the venv works only inside Apptainer)" >&2; exit 1; }
task_env=(OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
          VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
          CUDA_VISIBLE_DEVICES="" "PYTHONPATH=$task_dir:$task_base"
          "ROUTEGATE_DATA_DIR=${ROUTEGATE_DATA_DIR:-$HOME/routegate-data}")
if [ "$task_stage" = selftest ]; then
  apptainer exec --cleanenv --bind "$HOME:$HOME" "$task_image" \
    /usr/bin/env "${task_env[@]}" "$task_python" "$task_dir/test_moe.py"
else
  apptainer exec --cleanenv --bind "$HOME:$HOME" "$task_image" \
    /usr/bin/env "${task_env[@]}" "$task_python" "$task_dir/run_moe.py" "$task_stage" --repo "$task_repo" "$@"
fi
